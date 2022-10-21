from gym_collision_avoidance.envs.config import Config

# 配置文件
class Train(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 1
        self.MAX_NUM_AGENTS_TO_SIM = 1
        self.TRAIN_SINGLE_AGENT = True
        super().__init__()
        self.STATES_IN_OBS = ['dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius']
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.SAVE_EPISODE_PLOTS = False
        self.EVALUATE_MODE = False
        self.TRAIN_MODE = True
        self.DT = 0.2
        self.PLOT_CIRCLES_ALONG_TRAJ = True

class Test(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 19
        # self.TRAIN_SINGLE_AGENT = True
        super().__init__()
        self.DT = 0.2
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.STATES_IN_OBS = ['dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius']
        self.SAVE_EPISODE_PLOTS = True
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATE_EPISODES = True
        self.SHOW_EPISODE_PLOTS = True
