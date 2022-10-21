from cgi import print_environ
import os
import gym
import numpy as np
gym.logger.set_level(40)
os.environ['GYM_CONFIG_PATH'] = "emulation_config.py"
os.environ['GYM_CONFIG_CLASS'] = 'Test'

from gym_collision_avoidance.model import set_agents
from gym_collision_avoidance.model.visualize_plot import plot_episode

def main():

    env = gym.make("CollisionAvoidance-v0")

    # env.set_plot_save_dir(
    #     os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/example/')
    agents = set_agents.generate_random_human_position(num_agents= 10,circle_radius=10)
    env.set_agents(agents)
    obs = env.reset()
    num_steps = 300

    for i in range(num_steps):
        actions = {}
        actions[0] = np.array([2])
        obs, rewards, game_over, which_agents_done = env.step(actions)
        # print(which_agents_done)
        # for key, value in which_agents_done.items():
            # print(f"key:{key}, value:{value}")

        if game_over:
                print("All agents finished!")
                break
    
    env.reset()
    print(f"steps = {i}")

if __name__ == '__main__':
    main()
    
    print("Experiment over.")





