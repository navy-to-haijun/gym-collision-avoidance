import numpy as np
from numpy.linalg import norm
from gym_collision_avoidance.envs.agent import Agent
# policy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
# 自己policy
from gym_collision_avoidance.model.learningpolicytest import LearningPolicyTest

# Dynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.dynamics.ExternalDynamics import ExternalDynamics
# Sensors
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
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
                    agents.append(Agent(px,py,gx,gy,radius,v_pref,-(np.pi - angle),CADRLPolicy,UnicycleDynamics,[OtherAgentsStatesSensor], i))
                else:
                    agents.append(Agent(px,py,gx,gy,radius,v_pref,-(np.pi - angle),RVOPolicy,UnicycleDynamics,[OtherAgentsStatesSensor], i))
                break
    return agents

def set_agents_2():
    agents = [
        Agent(0., 6., 12., 6., 0.35, 0.5, 0.0, CADRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
        Agent(12.,6.,0.,6.,0.35,0.5, 0, RVOPolicy, UnicycleDynamics,[OtherAgentsStatesSensor], 0),
    ] 
    return agents

# def set_agent_2():
#     agents = [
#         Agent(0., 6., 12., 6, 0.35, 0.5, 0, GA3CCADRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
#         Agent(3, 9., 9, 3, 0.4, 0.5, 3*np.pi/4, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1),
#         Agent(3, 3, 9, 9, 0.4, 0.5, -np.pi/4, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 2),
#     ]
#     return agents

# def set_agent_4():
#     agents = [
#         Agent(0., 6., 12., 6, 0.35, 0.5, 0, CADRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
#         Agent(2, 4, 10, 8, 0.4, 0.5, np.pi/2, GA3CCADRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1),
#         Agent(4, 2, 8, 10, 0.4, 0.5, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 2),
#         Agent(4, 10, 8, 2, 0.4, 0.6, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 3),
#         Agent(2, 8,10, 4,  0.4, 0.4, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 4),
#     ]
#     return agents

# def set_agent_6():
#     agents = [
#         Agent(0., 6., 12., 6, 0.35, 0.5, np.pi/2, GA3CCADRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
#         Agent(1, 5., 11, 7, 0.4, 0.5, 3*np.pi/4, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1),
#         Agent(2, 4, 10, 8, 0.4, 0.5, np.pi/2,  RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 2),
#         Agent(3, 3, 9, 9, 0.4, 0.4, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 3),
#         Agent(1, 7, 11, 5, 0.4, 0.8, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 4),
#         Agent(2, 8, 10, 4,  0.4, 0.6, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 5),
#         Agent(3, 9, 9, 3,  0.4, 0.6, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 6),
#     ]
#     return agents

# def set_agent_6():
#     agents = [
#         Agent(0., 6., 12., 6, 0.35, 0.5, np.pi/2, GA3CCADRLPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
#         Agent(12, 6., 0, 6, 0.4, 0.5, 3*np.pi/4, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1),
#         Agent(2, 4, 10, 8, 0.4, 0.5, np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 2),
#         Agent(4, 2, 8, 10, 0.4, 0.4, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 3),
#         Agent(4, 10, 8, 2, 0.4, 0.8, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 4),
#         Agent(2, 8,10, 4,  0.4, 0.6, -np.pi/2, RVOPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 5),
    # ]
    # return agents