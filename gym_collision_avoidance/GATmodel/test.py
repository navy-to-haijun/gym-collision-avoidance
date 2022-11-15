
from operator import mod
import os
from statistics import mode
import gym
import numpy as np
import torch

import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from ppornn.ppo_discrete_rnn import PPO_discrete_RNN

# 仿真环境
os.environ['GYM_CONFIG_PATH'] = "emulation_config.py"
os.environ['GYM_CONFIG_CLASS'] = 'Train'
gym.logger.set_level(40)
import argparse
import ppo_util
def main(args):
    env_name = "CollisionAvoidance-v0"
    env = ppo_util.create_env(env_name)

    agents = ppo_util.generate_random_human_position(radius=0.3)
    # agents = ppo_util.generate_human_position()
    env.set_agents(agents)
    obs = env.reset()
    fov_obs = ppo_util.limit_distance_FOV(obs, FOV=180)
    s = fov_obs

    args.state_dim = 16
    args.action_dim = 11
    args.episode_limit = 100

    # for _ in range(25):
    #     actions = {}
    #     actions[0] = 2
    #     obs, rewards, game_over, which_agents_done = env.step(actions)
    #     fov_obs = ppo_util.limit_distance_FOV(obs, FOV=180, distance=3)
    #     s = np.vstack((s, fov_obs))
    #     print(f'human = {fov_obs[4]}')

    # from visualize_plot import train_plot
    # # train_plot(agents,fov=180)

    model = PPO_discrete_RNN(args)
    s = torch.tensor(s,dtype=torch.float32)
    s = s.unsqueeze(0).unsqueeze(0)
    # print(s.shape)
    # s = s.reshape(2,-1,110)

    x, robot_features_list = model.batch_graph_data(s)
    # print(x.shape)
    # for i in range(len(data.ptr)):
    #     human = torch.index_select(x, 0, [data.ptr[i]:data.ptr[i+1]])
    # print(f'x={batch.ptr}')
    # fov_obs = np.vstack((fov_obs, fov_obs))
    # print(s[0][1])
    # data, origin_robot = model.states_to_graph(fov_obs)
    # print(data)
    # print(data.x)
    # print(data.edge_index)
    # x= model.test_gat(data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting")
    parser.add_argument("--max_train_steps", type=int, default=int(2e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--evaluate_times", type=int, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="Minibatch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=False, help="Trick 10: tanh activation function")
    parser.add_argument("--use_gru", type=bool, default=True, help="Whether to use GRU")
    parser.add_argument("--log_name", type=str, default="ppornn_test",help="store log files")
    parser.add_argument("--retrain",type=bool, default=False, help="continue train")
    parser.add_argument("--evaluate",type=bool, default=False, help="continue train")

    args = parser.parse_args()
    main(args)


