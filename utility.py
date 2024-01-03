import numpy as np


def agentObserv(state, position, size):
    sliced_image = np.zeros(size).astype(np.uint8)

    row = (size[0] - 1) // 2
    col = (size[1] - 1) // 2

    for ri, dr in enumerate(np.arange(-row, row + 1, 1)):
        for ci, dc in enumerate(np.arange(-col, col + 1, 1)):
            r = position[0] + dr
            c = position[1] + dc

            if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                sliced_image[ri, ci] = state[r, c]
    return sliced_image


def evaluateActions(position, actions):
    positions = []
    x, y = position
    positions.append((x, y))
    for a in actions:
        x, y = positions[-1]
        if a == 0:
            positions.append((x, y))
        elif a == 1:
            positions.append((x - 1, y + 1))
        elif a == 2:
            positions.append((x, y + 1))
        elif a == 3:
            positions.append((x + 1, y + 1))
        elif a == 4:
            positions.append((x - 1, y))
        elif a == 5:
            positions.append((x + 1, y))
        elif a == 6:
            positions.append((x - 1, y - 1))
        elif a == 7:
            positions.append((x, y - 1))
        elif a == 8:
            positions.append((x + 1, y - 1))

    return positions


def rc2xy(height, rc):
    return rc[1] + 1, height - rc[0]


def xy2rc(height, xy):
    return height - xy[1], xy[0] - 1


def moveCenter(agent):
    distances = []
    for idx, a in enumerate([2, 5, 7, 4, 1, 3, 8, 6]):
        new_position = evaluateActions(agent.position, [a])[1]
        incentive = (
            -(8 - idx) * 0.1
        )  # bias choice towards certain order of agent actions
        d = np.linalg.norm(agent.fire_center - np.asarray(new_position), 1) + incentive
        distances.append((d, a))

    _, action = min(distances, key=lambda t: t[0])
    return action


def reward(forest_state, agent):
    total_reward = 0

    x1, y1 = agent.position
    x2, y2 = agent.next_position

    r_image = -y2 + y1 + agent.size[1] // 2
    c_image = x2 - x1 + agent.size[0] // 2

    r_forest, c_forest = xy2rc(forest_state.shape[1], agent.next_position)

    if agent.image[r_image, c_image] == agent.on_fire:
        healthy_neighbors = 0
        for dr, dc in agent.fire_neighbors:
            rn, cn = r_forest + dr, c_forest + dc
            if (
                rn < 0
                or rn >= forest_state.shape[1]
                or cn < 0
                or cn >= forest_state.shape[0]
            ):
                healthy_neighbors += 1
            elif forest_state[rn, cn] == agent.healthy:
                healthy_neighbors += 1

        if healthy_neighbors > 0:
            total_reward += 1
        else:
            total_reward += -2

    elif agent.image[r_image, c_image] == agent.healthy:
        non_healthy_numbers = 0
        for dr, dc in agent.move_deltas:
            rn, cn = r_forest + dr, c_forest + dc
            if (
                rn < 0
                or rn >= forest_state.shape[1]
                or cn < 0
                or cn >= forest_state.shape[0]
            ):
                continue
            elif forest_state[rn, cn] in [agent.on_fire, agent.burnt]:
                non_healthy_numbers += 1

        if non_healthy_numbers > 0:
            total_reward += 0.5
        else:
            total_reward += -1

    if agent.numeric_id > agent.closest_agent_id:
        if np.linalg.norm(agent.next_position - agent.closest_agent_position, 2) <= 1:
            total_reward += -10
        elif (
            np.linalg.norm(agent.position - agent.closest_agent_position, 2)
            <= 1
            < np.linalg.norm(agent.next_position - agent.closest_agent_position, 2)
        ):
            total_reward += 1

    move_vector = agent.next_position - agent.position
    norm = np.linalg.norm(move_vector, 2)
    if norm != 0:
        move_vector = move_vector / norm

    center_vector = agent.position - agent.fire_center
    norm = np.linalg.norm(center_vector, 2)
    if norm != 0:
        center_vector = center_vector / norm

    rotation_score = -1 * np.cross(center_vector, move_vector)
    if rotation_score >= 0:
        total_reward += 1

    return total_reward


def heuristic(agent):
    action = None

    if agent.reached_fire:
        distances = []
        fire_center_vector = agent.position - agent.fire_center
        norm = np.linalg.norm(fire_center_vector, 2)
        if norm != 0:
            fire_center_vector = fire_center_vector / norm

        for a in range(1, 9):
            new_position = evaluateActions(agent.position, [a])[1]
            move_vector = np.asarray(new_position) - agent.position
            move_vector = move_vector / np.linalg.norm(move_vector, 2)
            distances.append(
                (np.cross(fire_center_vector, move_vector), new_position, a)
            )

        _, circular_position, action = min(distances, key=lambda t: t[0])

        ri = -circular_position[1] + agent.position[1] + (agent.size[0] - 1) // 2
        ci = circular_position[0] - agent.position[0] + (agent.size[1] - 1) // 2

        left_action = None
        right_action = None
        if action == 1:
            left_action = [6, 4]
            right_action = 2
        elif action == 2:
            left_action = [4, 1]
            right_action = 3
        elif action == 3:
            left_action = [1, 2]
            right_action = 5
        elif action == 5:
            left_action = [2, 3]
            right_action = 8
        elif action == 8:
            left_action = [3, 5]
            right_action = 7
        elif action == 7:
            left_action = [5, 8]
            right_action = 6
        elif action == 6:
            left_action = [8, 7]
            right_action = 4
        elif action == 4:
            left_action = [7, 6]
            right_action = 1

        move_left = False
        for a in left_action:
            new_position = evaluateActions(agent.position, [a])[1]
            ro = -new_position[1] + agent.position[1] + (agent.size[0] - 1) // 2
            co = new_position[0] - agent.position[0] + (agent.size[1] - 1) // 2
            if agent.image[ro, co] == agent.on_fire:
                circular_position = new_position
                action = a
                move_left = True
                break

        if not move_left:
            for a in left_action:
                new_position = evaluateActions(agent.position, [a])[1]
                ro = -new_position[1] + agent.position[1] + (agent.size[0] - 1) // 2
                co = new_position[0] - agent.position[0] + (agent.size[1] - 1) // 2
                if agent.image[ro, co] == agent.burnt:
                    circular_position = new_position
                    action = a
                    move_left = True
                    break

        if not move_left and agent.image[ri, ci] == agent.healthy:
            healthy_neighbors = 0
            for dr, dc in agent.move_deltas:
                rn, cn = ri + dr, ci + dc
                if 0 <= rn < agent.size[1] and 0 <= cn < agent.size[0]:
                    if agent.image[rn, cn] == agent.healthy:
                        healthy_neighbors += 1
                else:
                    healthy_neighbors += 1

            if healthy_neighbors >= 6:
                circular_position = evaluateActions(agent.position, [right_action])[1]
                action = right_action

        if (
            np.linalg.norm(circular_position - agent.closest_agent_position, 2) <= 1
            and agent.numeric_id > agent.closest_agent_id
        ):
            action = 0
    else:
        action = moveCenter(agent)

    return action


def getAgentPositions(team):
    return [i.position for i in team.values()]
