import torch
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
gym.logger.set_level(40)
import argparse
# ppo 算法
from ppo.normalization import Normalization, RewardScaling
from ppo.replaybuffer import ReplayBuffer
from ppo.ppo_discrete import PPO_discrete
# 仿真环境
os.environ['GYM_CONFIG_PATH'] = "emulation_config.py"
os.environ['GYM_CONFIG_CLASS'] = 'Train'
import ppo_util

# 评估
def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    dist_to_goal = 0
    for _ in range(times):
        agents = ppo_util.generate_random_human_position(num_agents=1)
        env.set_agents(agents)
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            # 动作转化
            actions = {}
            actions[0] = np.array([a])
            s_, r, done, which_agents_done = env.step(actions)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward
        dist_to_goal += agents[0].get_agent_data("dist_to_goal")

    return evaluate_reward / times, dist_to_goal/times

def main(args):
    # 加载环境
    env_name = "CollisionAvoidance-v0"
    env = ppo_util.create_env(env_name)
    env_evaluate = ppo_util.create_env(env_name) # 评估使用
    agents = ppo_util.generate_random_human_position(num_agents=1)
    env.set_agents(agents)
    env_evaluate.set_agents(agents)
    #设置状态空间、动作空间
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = 11
    args.max_episode_steps = 100
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))

    # log 存储位置
    directory = "ppo_log"
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory = directory + '/'+ args.log_name+ '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint_path = directory  + "ppo_env_{}.pth".format(env_name)
    tensorboard_path = directory + "ppo_env_{}".format(env_name)

    evaluate_num = 0                     # 记录评估的次数
    total_steps = 0                      # 记录训练中的步数

    replay_buffer = ReplayBuffer(args)   # 存储序列
    ppo_policy = PPO_discrete(args)      # ppo policy

    # 断点训练
    if args.retrin | args.evaluate :
        total_steps = ppo_policy.load(args, checkpoint_path)
    # Build a tensorboard
    writer = SummaryWriter(log_dir= tensorboard_path)

    state_norm = Normalization(shape=args.state_dim)  # 状态归一化
    # 处理奖励：归一化 or 缩放
    if args.use_reward_norm:                         
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    
    # 开始训练
    while total_steps < args.max_train_steps:
        # 初始化环境
        episode_steps = 0
        agents = ppo_util.generate_random_human_position(num_agents=1)
        env.set_agents(agents)
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = ppo_policy.choose_action(s) # 产生动作
            # 动作转化
            actions = {}
            actions[0] = np.array([a])
            s_, r, done, which_agents_done = env.step(actions)
            if episode_steps == args.max_episode_steps:
                done = True

            if args.use_state_norm:
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)
            
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False
            # 存储轨迹
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = s_
            total_steps += 1
            # 更新 policy
            if replay_buffer.count == args.batch_size:
                ppo_policy.update(replay_buffer, total_steps, writer)
                replay_buffer.count = 0
            # 评估 每隔evaluate_freq 计算一次奖励，每隔evaluate_freq * save_freq 保存 网络参数
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, dist_to_goal = evaluate_policy(args, env_evaluate, ppo_policy, state_norm)
                print("total_steps:{} \t reward:{} \t goal:{}".format(total_steps, evaluate_reward[0], dist_to_goal))
                # 记录
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_reward[0], global_step=total_steps)
                # Save the checkpoint
                if evaluate_num % args.save_freq == 0:
                    ppo_policy.save(checkpoint_path, total_steps)
                    print("episode_steps={}\t saving model at: {} ".format(episode_steps, checkpoint_path))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=10, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--log_name", type=str, default="ppo_test",help="store log files")
    parser.add_argument("--retrin",type=bool, default=False, help="continue train")
    parser.add_argument("--evaluate",type=bool, default=False, help="continue train")

    args = parser.parse_args()
    main(args)