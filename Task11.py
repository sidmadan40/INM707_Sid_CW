import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
import matplotlib.pyplot as plt
from gymnasium.utils import EzPickle
import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.classic_control import PendulumEnv


import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import seaborn as sns

class SimpleCorridor(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config=None):
        config = config or {}
        self.end_pos = config.get("corridor_length", 10)
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, 999.0, shape=(1,), dtype=np.float32)

    def set_corridor_length(self, length):
        self.end_pos = length
        print("Updated corridor length to {}".format(length))

    def reset(self, *, seed=None, options=None):
        self.cur_pos = 0.0
        return [self.cur_pos], {}


    def step(self, action):
        assert action in [0, 1], action
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1.0
        elif action == 1:
            self.cur_pos += 1.0
        done = truncated = self.cur_pos >= self.end_pos
        reward = -0.1  # negative reward for each action taken
        if done:
            reward += 1  # bonus reward for reaching the end of the corridor
        return [self.cur_pos], reward, done, truncated, {}

config = (
    PPOConfig()
    .environment(
        env=SimpleCorridor,
    )
    .rollouts(num_rollout_workers=3)
)
algo = config.build()
average_rewards = []

for i in range(100):
    results = algo.train()
    average_rewards.append(results['episode_reward_mean'])
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")


sns.set_style("darkgrid")
sns.set_palette("husl")

# Plot the average rewards
plt.figure(figsize=(10,6))
plt.plot(average_rewards)
plt.xlabel('Number of Iteration')
plt.ylabel('Iterated average Reward')
plt.title('Algorithm Performance')
plt.show()