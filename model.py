import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from collections import defaultdict
import copy
import numpy as np
import time
from simulators.fires.LatticeForest import LatticeForest

from utility import (
    agentObserv,
    evaluateActions,
    xy2rc,
    moveCenter,
    reward,
    heuristic,
    getAgentPositions,
)
from simulator import Simulator


class Config:
    def __init__(self, config_type="train"):
        self.dtype = torch.float64
        self.forest_dimension = 50
        self.delta_beta = 0.15 / 0.2763
        self.forest_iter_limit = 1000
        self.size = (3, 3)
        self.fire_center = np.array(
            [(self.forest_dimension + 1) / 2, (self.forest_dimension + 1) / 2]
        )
        self.update_sim_every = 6
        self.update_forest_every = 1

        # agent initialization locations
        self.start = np.arange(
            self.forest_dimension // 3 // 2,
            self.forest_dimension,
            self.forest_dimension // 3,
        )
        self.perturb = np.arange(
            -self.forest_dimension // 3 // 2 + 1, self.forest_dimension // 3 // 2 + 1, 1
        )

        # network input parameters
        self.feature_length = np.prod(self.size) + 2 + 1 + 2
        self.action_length = 1
        self.reward_length = 1

        if config_type == "train":
            self.save_directory = "/checkpoints"

            # simulation iteration cutoff
            self.sim_iter_limit = 100

            # replay memory
            self.memory_size = 1000000
            self.min_experience_size = 5000

            # target network instance
            self.update_target_every = 6000  # 6000

            # optimizer
            self.gamma = 0.95
            self.batch_size = 32
            self.learning_rate = 1e-4

            # exploration
            self.eps_ini = 1
            self.eps_fin = 0.15
            self.anneal_range = 20000  # 40000

            # loss function
            self.loss_fn = nn.MSELoss(reduction="mean")

        elif config_type == "test":
            self.base_station = np.array([5, 5])


class UAV:
    healthy = 0
    on_fire = 1
    burnt = 2

    move_deltas = [
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (-1, -1),
        (0, -1),
        (1, -1),
    ]  # excluded (0,0)
    fire_neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1)]

    def __init__(
        self,
        numeric_id=None,
        initial_position=None,
        fire_center=None,
        size=(3, 3),
    ):
        self.numeric_id = numeric_id
        self.initial_position = initial_position
        self.fire_center = fire_center
        self.size = size

        self.position = self.initial_position
        self.next_position = None

        self.capacity = None

        self.image = None
        self.reached_fire = False
        self.rotation_vector = None
        self.closest_agent_id = None
        self.closest_agent_position = None
        self.closest_agent_vector = None

        self.features = None

        self.actions = []
        self.rewards = []

    def reset(self):
        self.reached_fire = False
        self.next_position = None
        self.actions = []
        self.rewards = []

    def update_position(self):
        self.position = self.next_position
        self.next_position = None

    def update_features(self, forest_state, team):
        height, width = forest_state.shape
        self.image = agentObserv(forest_state, xy2rc(height, self.position), self.size)
        image_center = (self.size[0] - 1) // 2, (self.size[1] - 1) // 2
        if self.image[image_center[0], image_center[1]] in [self.on_fire, self.burnt]:
            self.reached_fire = True

        self.rotation_vector = self.position - self.fire_center
        norm = np.linalg.norm(self.rotation_vector, 2)
        if norm != 0:
            self.rotation_vector = self.rotation_vector / norm
        self.rotation_vector = np.array(
            [self.rotation_vector[1], -self.rotation_vector[0]]
        )

        d = [
            (
                np.linalg.norm(self.position - agent.position, 2),
                agent.numeric_id,
                agent.position,
            )
            for agent in team.values()
            if agent.numeric_id != self.numeric_id
        ]
        _, self.closest_agent_id, self.closest_agent_position = min(
            d, key=lambda x: x[0]
        )

        self.closest_agent_vector = self.position - self.closest_agent_position
        norm = np.linalg.norm(self.closest_agent_vector)
        if norm != 0:
            self.closest_agent_vector = self.closest_agent_vector / norm

        return np.concatenate(
            (
                self.image.ravel(),
                self.rotation_vector,
                np.asarray(self.numeric_id > self.closest_agent_id)[np.newaxis],
                self.closest_agent_vector,
            )
        )


class NeuralNetwork(nn.Module):
    def __init__(self, size=(3, 3)):
        super(NeuralNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(size[0] * size[1] + 2 + 1 + 2, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 9),
        )

    def forward(self, features):
        return self.network(features)


class MODEL:
    def __init__(self, mode="train", filename=None):
        self.mode = mode
        self.config = Config(config_type=self.mode)
        self.model = NeuralNetwork(size=self.config.size).type(torch.float64)

        if mode == "train":
            self.target = NeuralNetwork(size=self.config.size).type(torch.float64)
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate
            )
            self.sars = None
            self.eps = self.config.eps_ini
            self.reward_history = []
            self.loss_history = []
            self.num_train_episodes = 0
            self.print_enough_experiences = False

        if filename is not None:
            self.load_checkpoint(filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["state_dict"])

        if self.mode == "train":
            self.target.load_state_dict(checkpoint["target_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.sars = checkpoint["replay"]
            self.eps = checkpoint["epsilon"]
            self.reward_history = checkpoint["reward_history"]
            self.loss_history = checkpoint["loss_history"]
            self.num_train_episodes = checkpoint["num_train_episodes"]

    def save_checkpoint(self):
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "target_dict": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.sars,
            "epsilon": self.eps,
            "reward_history": self.reward_history,
            "loss_history": self.loss_history,
            "num_train_episodes": self.num_train_episodes,
        }
        filename = "madqn-" + time.strftime("%d-%b-%Y-%H%M") + ".pth.tar"
        torch.save(checkpoint, filename)

    def train(self, num_episodes=1):
        forest = LatticeForest(self.config.forest_dimension)
        num_agents = 10
        team = {
            i: UAV(numeric_id=i, fire_center=self.config.fire_center)
            for i in range(num_agents)
        }

        model_updates = 1
        metrics = []
        print("Completed(%):", end="")
        for episode in range(num_episodes):
            print("#", end="")
            stats = {}
            forest.reset()
            for agent in team.values():
                agent.reset()
                agent.position = np.random.choice(
                    self.config.start, 2
                ) + np.random.choice(self.config.perturb, 2)
                agent.initial_position = agent.position
            sim_updates = 1
            sim_control = defaultdict(lambda: (0.0, 0.0))
            forest_updates = 1
            forest_control = defaultdict(lambda: (0.0, 0.0))

            while not forest.end:
                forest_state = forest.dense_state()
                for agent in team.values():
                    agent.features = agent.update_features(forest_state, team)

                    action = None
                    # move over fire if not on the fire
                    if not agent.reached_fire:
                        action = moveCenter(agent)
                    else:
                        # explore
                        if np.random.rand() <= self.eps:
                            action = heuristic(agent)
                        # exploit
                        else:
                            q_features = Variable(
                                torch.from_numpy(agent.features)
                            ).type(self.config.dtype)
                            q_values = (
                                self.model(q_features.unsqueeze(0))[0]
                                .data.cpu()
                                .numpy()
                            )
                            action = np.argmax(q_values)

                    agent.actions.append(action)
                    agent.next_position = np.asarray(
                        evaluateActions(agent.position, [action])[1]
                    )
                    agent.rewards.append(reward(forest_state, agent))

                for agent in team.values():
                    agent.update_position()
                    position_rc = xy2rc(forest.dims[0], agent.position)
                    if (
                        0 <= position_rc[0] < forest.dims[0]
                        and 0 <= position_rc[1] < forest.dims[1]
                    ):
                        if forest.group[position_rc].is_on_fire(
                            forest.group[position_rc].state
                        ):
                            forest_control[position_rc] = (0.0, self.config.delta_beta)

                if forest_updates % self.config.update_forest_every == 0:
                    forest.update(forest_control)
                    forest_control = defaultdict(lambda: (0.0, 0.0))
                    forest_state = forest.dense_state()

                forest_updates += 1

                if forest.end:
                    continue

                for agent in team.values():
                    if not agent.reached_fire:
                        continue
                    next_features = agent.update_features(forest_state, team)
                    data = np.zeros(
                        (
                            1,
                            2 * self.config.feature_length
                            + self.config.action_length
                            + self.config.reward_length,
                        )
                    )
                    data[0, 0 : self.config.feature_length] = agent.features
                    data[0, self.config.feature_length] = agent.actions[-1]
                    reward_idx = self.config.feature_length + self.config.action_length
                    data[0, reward_idx] = agent.rewards[-1]
                    next_features_idx = (
                        self.config.feature_length
                        + self.config.action_length
                        + self.config.reward_length
                    )
                    data[0, next_features_idx:] = next_features

                    if self.sars is None:
                        self.sars = data
                    else:
                        self.sars = np.vstack((self.sars, data))
                if (
                    self.sars is None
                    or self.sars.shape[0] < self.config.min_experience_size
                    or self.sars.shape[0] < self.config.batch_size
                ):
                    continue
                elif not self.print_enough_experiences:
                    self.print_enough_experiences = True
                batch = self.sars[
                    np.random.choice(
                        self.sars.shape[0], self.config.batch_size, replace=False
                    ),
                    :,
                ]
                batch_features = torch.from_numpy(
                    batch[:, 0 : self.config.feature_length]
                )
                batch_features = Variable(batch_features).type(self.config.dtype)
                batch_actions = torch.from_numpy(batch[:, self.config.feature_length])
                batch_actions = Variable(batch_actions).type(torch.int64)
                x = (
                    self.model(batch_features)
                    .gather(1, batch_actions.view(-1, 1))
                    .squeeze()
                )
                batch_rewards = batch[
                    :, self.config.feature_length + self.config.reward_length
                ]
                next_features_idx = (
                    self.config.feature_length
                    + self.config.action_length
                    + self.config.reward_length
                )
                batch_next_features = Variable(
                    torch.from_numpy(batch[:, next_features_idx:])
                ).type(self.config.dtype)
                tt = self.target(batch_next_features).data.cpu().numpy()
                tt = batch_rewards + self.config.gamma * np.amax(tt, axis=1)
                tt = Variable(torch.from_numpy(tt), requires_grad=False).type(
                    self.config.dtype
                )
                loss = self.config.loss_fn(x, tt)

                self.loss_history.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.eps > self.config.eps_fin:
                    self.eps += (
                        -(self.config.eps_ini - self.config.eps_fin)
                        / self.config.anneal_range
                    )
                if model_updates % self.config.update_target_every == 0:
                    self.target = copy.deepcopy(self.model)
                model_updates += 1
                if self.sars.shape[0] > self.config.memory_size:
                    self.sars = self.sars[
                        self.sars.shape[0] - self.config.memory_size, :
                    ]

            stats = {}
            stats["percent_healthy"] = forest.getfracHealthy()
            stats["reward_per_agent"] = np.mean(
                [np.mean(agent.rewards) for agent in team.values()]
            )
            metrics.append(stats)
            self.reward_history.append(stats["reward_per_agent"])
            self.num_train_episodes += 1
        self.save_checkpoint()
        return metrics

    def test(self, num_episodes=1, num_agents=10, capacity=None):
        forest = LatticeForest(self.config.forest_dimension)
        team = {
            i: UAV(numeric_id=i, fire_center=self.config.fire_center)
            for i in range(num_agents)
        }

        # returns params for each episode in a list
        # for each episode we have a dict with percentage_healty , total_reward_per_agent
        metrics = []
        # iterating over episodes
        for episode in range(num_episodes):
            # reseting forest and agents
            np.random.seed(episode)
            forest.rng = episode
            forest.reset()

            # redeploying agents randomly
            for agent in team.values():
                agent.reset()
                agent.position = np.random.choice(
                    self.config.start, 2
                ) + np.random.choice(self.config.perturb, 2)
                agent.initial_position = agent.position

                if capacity is not None:
                    agent.capacity = capacity

            sim_updates = 1
            sim_control = defaultdict(lambda: (0.0, 0.0))

            forest_state = forest.dense_state()
            sim_states = [forest_state]

            size = (self.config.forest_dimension, self.config.forest_dimension)
            noAgent = len(getAgentPositions(team))
            simulator = Simulator(size, noAgent)

            while not forest.end:
                print(getAgentPositions(team))
                simulator.draw_trees(forest.getState())
                simulator.draw_agent(getAgentPositions(team))

                for agent in team.values():
                    agent.features = agent.update_features(forest_state, team)

                    action = None
                    if not agent.reached_fire:
                        action = moveCenter(agent)
                    else:
                        q_features = Variable(torch.from_numpy(agent.features)).type(
                            self.config.dtype
                        )
                        q_values = (
                            self.model(q_features.unsqueeze(0))[0].data.cpu().numpy()
                        )
                        action = np.argmax(q_values)

                    agent.actions.append(action)
                    agent.next_position = np.asarray(
                        evaluateActions(agent.position, [action])[1]
                    )
                for agent in team.values():
                    agent.update_position()
                    position_rc = xy2rc(forest.dims[0], agent.position)
                    if (
                        0 <= position_rc[0] < forest.dims[0]
                        and 0 <= position_rc[1] < forest.dims[1]
                    ):
                        if forest.group[position_rc].is_on_fire(
                            forest.group[position_rc].state
                        ):
                            sim_control[position_rc] = (0.0, self.config.delta_beta)

                            if capacity is not None:
                                agent.capacity -= 1
                    if capacity is not None and agent.capacity == 0:
                        agent.reached_fire = False
                        agent.next_position = None
                        agent.position = self.config.base_station

                if sim_updates % self.config.update_sim_every == 0:
                    forest.update(sim_control)
                    sim_control = defaultdict(lambda: (0.0, 0.0))
                    forest_state = forest.dense_state()

                sim_updates += 1
                sim_states.append(forest_state)

            stats = {}
            stats["percent_healthy"] = forest.getfracHealthy()
            stats["reard_per_agent"] = np.mean(
                [np.mean(agent.rewards) for agent in team.values()]
            )
            metrics.append(stats)
        print("Done...")
        simulator.quit()
        return metrics
