from model import MODEL
import matplotlib.pyplot as plt

model = MODEL(mode="test", filename="best_model.tar")
# metrics = algorithm.train(num_episodes=5)

# percent_healthy = []
# for episode in metrics:
#     percent_healthy.append(episode["percent_healthy"])

# plt.plot(percent_healthy)
# plt.xlabel("Episode")
# plt.ylabel("Percent reamaing trees")
# plt.savefig("Percent.png")

# rewards = []
# for episode in metrics:
#     rewards.append(episode["reward_per_agent"])

# plt.plot(rewards)
# plt.xlabel("Episode")
# plt.ylabel("Reward per agent")
# plt.savefig("Reward.png")

# print(algorithm.test(num_episodes=1))
model.test()
