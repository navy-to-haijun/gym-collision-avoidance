import numpy as np
from numpy.linalg import norm
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
def generate_random_human_position(v_pref = 1, num_agents = 3, circle_radius=6, radius = 0.4, min_dist = 1.2):
    agents = []
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
                    agents.append(Agent(px,py,gx,gy,radius,v_pref,-(np.pi - angle),RVOPolicy,UnicycleDynamics,[OtherAgentsStatesSensor], i))
                break
    return agents

def create_env(name):
    # 创建单个环境
    env = gym.make(name)
    env = FlattenDictWrapper(env, dict_keys=Config.STATES_IN_OBS)

    return env