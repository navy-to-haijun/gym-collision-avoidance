import numpy as np
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy

# 实现 external_action_to_action()
# 原始速度为线速度和角速度,修改为11个离散速度

class Actions():
    def __init__(self):
        # Define 11 choices of actions to be:
        # [v_pref,      [-pi/6, -pi/12, 0, pi/12, pi/6]]
        # [0.5*v_pref,  [-pi/6, 0, pi/6]]
        # [0,           [-pi/6, 0, pi/6]]

        self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/12].reshape(2, -1).T
        self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/6:np.pi/6+0.01:np.pi/6].reshape(2, -1).T])
        self.num_actions = len(self.actions)

class LearningPolicyTest:
    def __init__(self):
        LearningPolicy.__init__(self)
        self.possible_actions = Actions()

    def external_action_to_action(self, agent, external_action):
        raw_action = self.possible_actions.actions[int(external_action)]
        action = np.array([agent.pref_speed*raw_action[0], raw_action[1]])
        return action