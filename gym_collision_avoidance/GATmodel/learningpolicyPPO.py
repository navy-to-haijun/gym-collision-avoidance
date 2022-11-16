import numpy as np
from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy

# 实现 external_action_to_action()
# 原始速度为线速度和角速度,修改为48个离散速度

class Actions():
    def __init__(self):
        # 离散动作
        self.actions = np.mgrid[0: 1: 6j, -np.pi/3: np.pi/3: np.pi/12].reshape(2, -1).T
        self.num_actions = len(self.actions)

class LearningPolicyPPO:
    def __init__(self):
        LearningPolicy.__init__(self)
        self.possible_actions = Actions()

    def external_action_to_action(self, agent, external_action):
        raw_action = self.possible_actions.actions[int(external_action)]
        action = np.array([agent.pref_speed*raw_action[0], raw_action[1]])
        return action