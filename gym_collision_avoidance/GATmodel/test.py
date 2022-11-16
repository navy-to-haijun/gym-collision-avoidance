import os
import gym
import numpy as np
from visualize_plot import train_plot
# 仿真环境
os.environ['GYM_CONFIG_PATH'] = "emulation_config.py"
os.environ['GYM_CONFIG_CLASS'] = 'Train'
gym.logger.set_level(40)
import ppo_util

env_name = "CollisionAvoidance-v0"
env = ppo_util.create_env(env_name)

agents = ppo_util.generate_random_human_position(radius=0.3)
# agents = ppo_util.generate_human_position()
env.set_agents(agents)
obs = env.reset()
fov_obs = ppo_util.limit_distance_FOV(obs, FOV=180)

for _ in range(50):
    actions = {}
    actions[0] = 44
    obs, rewards, game_over, which_agents_done = env.step(actions)
    fov_obs = ppo_util.limit_distance_FOV(obs, FOV=180, distance=3)
    train_plot(agents,fov=180)
