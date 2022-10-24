import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
gym.logger.set_level(40)
import argparse
# ppo 算法
from ppo.normalization import Normalization, RewardScaling
from ppornn.replaybuffer import ReplayBuffer
from ppornn.ppo_discrete_rnn import PPO_discrete_RNN
# 仿真环境
os.environ['GYM_CONFIG_PATH'] = "emulation_config.py"
os.environ['GYM_CONFIG_CLASS'] = 'Train'
import ppo_util

class Runner:
    def __init__(self, args):
        self.args = args
        self.env_name  = "CollisionAvoidance-v0"
        # Create env
        self.env = ppo_util.create_env(self.env_name)
        self.agents = ppo_util.generate_random_human_position(num_agents=1)
        self.env.set_agents(self.agents)

        #设置状态空间、动作空间
        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = 11
        self.args.episode_limit = 100

        print("env={}".format(self.env_name))
        print("state_dim={}".format(args.state_dim))
        print("action_dim={}".format(args.action_dim))
        print("episode_limit={}".format(args.episode_limit))


        # log 存储位置
        self.directory = "ppo_log"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.directory = self.directory + '/'+ args.log_name+ '/'
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.checkpoint_path = self.directory  + "ppo_env_{}.pth".format(self.env_name)
        self.tensorboard_path = self.directory + "ppo_env_{}".format(self.env_name)

        self.replay_buffer = ReplayBuffer(args)
        self.ppopolicy = PPO_discrete_RNN(args)
        
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.evaluate_num = -1

        # 断点训练
        if self.args.retrain | self.args.evaluate :
            self.total_steps, self.evaluate_num = self.ppopolicy.load(self.args, self.checkpoint_path) 
       
        # Create a tensorboard
        self.writer = SummaryWriter(log_dir=self.tensorboard_path)

        if self.args.use_state_norm:
            print("------use state normalization------")
            self.state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
        if self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=1, gamma=self.args.gamma)

    def run(self, ):
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > self.evaluate_num:
                self.evaluate_policy()  # 评估策略
                self.evaluate_num += 1

            _, episode_steps = self.run_episode()  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.ppopolicy.train(self.replay_buffer, self.total_steps, self.writer)  # 更新参数
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def run_episode(self, ):
        episode_reward = 0
        # 随机环境
        self.env = ppo_util.create_env(self.env_name)
        self.agents = ppo_util.generate_random_human_position(num_agents=1)
        self.env.set_agents(self.agents)
        s = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        self.ppopolicy.reset_rnn_hidden()
        for episode_step in range(self.args.episode_limit):
            if self.args.use_state_norm:
                s = self.state_norm(s)
            a, a_logprob = self.ppopolicy.choose_action(s, evaluate=False)
            v = self.ppopolicy.get_value(s)
            # 动作转化
            actions = {}
            actions[0] = np.array([a])
            s_, r, done, which_agents_done = self.env.step(actions)
            episode_reward += r

            if done and episode_step + 1 != self.args.episode_limit:
                dw = True
            else:
                dw = False
            if self.args.use_reward_scaling:
                r = self.reward_scaling(r)
            # Store the transition
            self.replay_buffer.store_transition(episode_step, s, v, a, a_logprob, r, dw)
            s = s_
            if done:
                break

        # An episode is over, store v in the last step
        if self.args.use_state_norm:
            s = self.state_norm(s)
        v = self.ppopolicy.get_value(s)
        self.replay_buffer.store_last_value(episode_step + 1, v)

        return episode_reward, episode_step + 1

    def evaluate_policy(self, ):
        evaluate_reward = 0
        dist_to_goal = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, done = 0, False
            # 随机环境
            self.env = ppo_util.create_env(self.env_name)
            self.agents = ppo_util.generate_random_human_position(num_agents=1)
            self.env.set_agents(self.agents)
            s = self.env.reset()
            self.ppopolicy.reset_rnn_hidden()
            while not done:
                if self.args.use_state_norm:
                    s = self.state_norm(s, update=False)
                a, a_logprob = self.ppopolicy.choose_action(s, evaluate=True)
                # 动作转化
                actions = {}
                actions[0] = np.array([a])
                s_, r, done, which_agents_done = self.env.step(actions)
                episode_reward += r
                s = s_
            evaluate_reward += episode_reward
            dist_to_goal += self.agents[0].get_agent_data("dist_to_goal")

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        dist_to_goal = dist_to_goal / self.args.evaluate_times

        self.writer.add_scalar('step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        print("total_steps:{} \t evaluate_reward:{} \t dist_to_goal:{}".format(self.total_steps, evaluate_reward, dist_to_goal))
        if self.evaluate_num % self.args.save_freq == 0:
            self.ppopolicy.save(self.checkpoint_path, self.total_steps, self.evaluate_num)
            print("episode_steps={}\t saving model at: {} ".format(self.total_steps, self.checkpoint_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
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

    runner = Runner(args)
    runner.run()
