from multiprocessing import Semaphore
import os
import gym
import numpy as np
os.environ['GYM_CONFIG_PATH'] = "config_test.py"
os.environ['GYM_CONFIG_CLASS'] = 'Train'

import threading
from threading import Thread, Semaphore

from gym_collision_avoidance.experiments.src.env_utils import create_env
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.model import agents_test
from gym_collision_avoidance.model.visualize_plot import plot_episode

env = gym.make("CollisionAvoidance-v0")

agents = agents_test.get_agents_3()
agents_copy = []
[agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
env.set_agents(agents)

# 信号量
semaphore = Semaphore(0)
num_steps = 5000

import copy
def main():
    obs = env.reset()
    episode_num = 0

    for i in range(num_steps):
        actions = {}
        actions[0] = np.array([2])
        obs, rewards, game_over, which_agents_done = env.step(actions)
         
        # 
        if game_over:
                episode_num  =episode_num +1
                print("All agents finished!")
                plot_episode(agents=agents,episode_num=episode_num ,plot_save_dir="./pictures/")
                env.reset()
                agents[0].reset(0., -20., 0., 10., 1.0, 0.5, np.pi/2)
                print(f"steps = {i}")
                
                # break
    
    env.reset()
    print(f"steps = {i}")

if __name__ == '__main__':
    main()
    
    print("Experiment over.")





