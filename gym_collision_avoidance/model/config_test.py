from gym_collision_avoidance.envs.config import Config

# 配置文件
class Train(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 3
        self.MAX_NUM_AGENTS_TO_SIM = 3
        self.TRAIN_SINGLE_AGENT = True
        super().__init__()
        self.STATES_IN_OBS = ['num_other_agents', 'dist_to_goal', 'heading_ego_frame', 'pref_speed', 'radius', 'other_agents_states']
        self.SHOW_EPISODE_PLOTS = False
        self.ANIMATE_EPISODES = False
        self.SAVE_EPISODE_PLOTS = False
        self.EVALUATE_MODE = False
        self.TRAIN_MODE = True
        self.DT = 0.1
        self.PLOT_CIRCLES_ALONG_TRAJ = True

# 