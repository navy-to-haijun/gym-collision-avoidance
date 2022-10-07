import numpy as np
from gym_collision_avoidance.envs.agent import Agent
# policy
from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
# 自己policy
from gym_collision_avoidance.model.learningpolicytest import LearningPolicyTest
# Dynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
# Sensors
from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor


def get_agents_3():
    agents = [
        Agent(0., -20., 0., 20., 0.5, 1.0, np.pi/2, LearningPolicyTest, UnicycleDynamics, [OtherAgentsStatesSensor], 0),
        Agent(-5, 0., -5, 5., 0.5, 1.0, np.pi/2, StaticPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 1),
        Agent(-5, 5, 5, -5, 0.5, 1.0, -np.pi/4, StaticPolicy, UnicycleDynamics, [OtherAgentsStatesSensor], 2),
    ] 
    return agents
