from gym_collision_avoidance.envs.config import Config

# 配置文件
class Train(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 16
        self.MAX_NUM_AGENTS_TO_SIM = 16
        self.TRAIN_SINGLE_AGENT = True
        super().__init__()
        self.STATES_IN_OBS = ['dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', "num_other_agents","other_agents_states"]
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.SAVE_EPISODE_PLOTS = False
        self.EVALUATE_MODE = False
        self.TRAIN_MODE = True
        self.DT = 0.2
        self.PLOT_CIRCLES_ALONG_TRAJ = False

class Test(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 10
        self.MAX_NUM_AGENTS_TO_SIM = 10
        # self.TRAIN_SINGLE_AGENT = True
        super().__init__()
        self.DT = 0.2
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        self.STATES_IN_OBS = ['dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', "num_other_agents","other_agents_states"]
        self.SAVE_EPISODE_PLOTS = True
        self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATE_EPISODES = True
        self.SHOW_EPISODE_PLOTS = True
