import numpy as np
from numpy.linalg import norm
import math
from gym_collision_avoidance.envs.agent import Agent
# policy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
# 自己policy
from learningpolicyPPO import LearningPolicyPPO
# Dynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.dynamics.ExternalDynamics import ExternalDynamics
# Sensors
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor

import gym
gym.logger.set_level(40)
import numpy as np
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.wrappers import FlattenDictWrapper
from gym_collision_avoidance.envs.vec_env import DummyVecEnv

# 设置agent
# 随机初始化agentsd的位置
def generate_random_human_position(max_agents = 16, v_pref = 1, circle_radius=6, radius = 0.3, min_dist = 1.0):
    agents = []
    num_agents = np.random.randint(3, max_agents+1)   # agent随机
    for i in range(num_agents):
        while True:
            angle = np.random.random() * np.pi * 2
            px_noise = (np.random.random() - 0.5) * v_pref
            py_noise = (np.random.random() - 0.5) * v_pref
            px = circle_radius * np.cos(angle) + px_noise
            py = circle_radius * np.sin(angle) + py_noise
            gx = -px
            gy = -py
            collide = False
            # agents之间保持一定距离
            for j in range(i):
                pos = agents[j].get_agent_data('pos_global_frame')
                if norm((px - pos[0], py - pos[1])) < min_dist:
                    collide = True
                    break
            if not collide:
                if i == 0:
                    agents.append(Agent(px,py,gx,gy,radius,v_pref,-(np.pi - angle),LearningPolicyPPO,UnicycleDynamics,[OtherAgentsStatesSensor], i))
                else:
                    # 行人为随机速度
                    agents.append(Agent(px,py,gx,gy,radius,np.random.uniform(0.5, 1.1),-(np.pi - angle),RVOPolicy,UnicycleDynamics,[OtherAgentsStatesSensor], i))
                break
    return agents

# 限制robot可观测到的行人
def limit_distance_FOV(obs,distance=3.0, FOV=360):
    num = 0
    new_obs = np.zeros(obs.shape)
    new_obs[0:4] = obs[0:4]  # robot 特征
    num_agents = int(obs[4]) # agents的数量
    robot_angle = math.degrees(obs[1])  # robot的朝向
    for i in range(5, num_agents*7, 7):
        dis = norm((obs[i], obs[i+1]))
        angle = math.degrees(math.atan2(obs[i+1],obs[i]))
        if dis < distance and ((robot_angle - FOV/2)<= angle<= (robot_angle + FOV/2)):
            new_obs[5+num*7:5+(num+1)*7] = obs[i:i+7]
            num+=1
    new_obs[4] = num
    return new_obs

def create_env(name):
    # 创建单个环境
    env = gym.make(name)
    env = FlattenDictWrapper(env, dict_keys=Config.STATES_IN_OBS)

    return env

# 临时函数
def generate_human_position():

    agents = [
        Agent(0, 0, 5, 5, 0.2, 1.0, 0.0, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
        Agent(-5, 5, 5, 5, 0.4, 0.9, 0, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1),
        Agent(5, 5, -5, -5, 0.3, 0.8, np.pi, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 2)
        ]
    return agents