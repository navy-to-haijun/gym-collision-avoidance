from ast import arg
from enum import Flag
from errno import ESTALE
from time import pthread_getcpuclockid
from tkinter.messagebox import NO
from traceback import print_tb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from torch.distributions import Categorical
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer
# 模型
class Actor_Critic_GAT_RNN(torch.nn.Module):
    def __init__(self, args):
        super(Actor_Critic_GAT_RNN, self).__init__()
        self.input_features = 4                 # 输入特征维度
        self.hid = 16                           # 隐藏层的维度
        self.in_head = 6                        # 注意力头数
        self.output_features = 32               # 输出特征维度
        self.out_head = 1
        self.fc1_hidden = 64
        self.rnn_hidden_dim = 64
        self.action_dim = 11
        self.actor_rnn_hidden = None
        self.critic_rnn_hidden = None
        self.activate_func = nn.Tanh()

        self.robot_features = 4
        self.human_features = 7
        # 特征嵌入
        self.robot_fc1 = nn.Linear(self.robot_features, 12)
        self.robot_fc2 = nn.Linear(12, 32)
        self.robot_fc3 = nn.Linear(32, 16)

        self.human_fc1 = nn.Linear(self.human_features, 16)
        self.human_fc2 = nn.Linear(16, 32)
        self.human_fc3 = nn.Linear(32, 16)

        # 注意力图
        self.gat_conv1 = GATConv(16, 32 ,heads=6, dropout=0.6)
        self.gat_conv2 = GATConv(32*6, 16, concat=False, dropout=0.6)

        # 行人特征聚合网络
        self.human_GRU = nn.GRU(16, 32, 2, batch_first=True)
        self.human_fc4 = nn.Linear(64, 16)
        self.human_rnn_hidden = None

        # actor
        self.actor_rnn = nn.GRU(36, 64, 2, batch_first=True)
        self.actor_fc1 = nn.Linear(64, 128)
        self.actor_fc2 = nn.Linear(128, 256)
        self.actor_fc3 = nn.Linear(256, 256)
        self.actor_fc4 = nn.Linear(256, 61)

        # critic
        self.critic_rnn = nn.GRU(36, 64, 2, batch_first=True)
        self.critic_fc1 = nn.Linear(64, 128)
        self.critic_fc2 = nn.Linear(128, 64)
        self.critic_fc3 = nn.Linear(64, 32)
        self.critic_fc4 = nn.Linear(32, 1)


    def feature_embeding(self,robot, human):
        robot = self.robot_fc1(robot)
        robot = self.activate_func(robot)
        robot = self.robot_fc2(robot)
        robot = self.activate_func(robot)
        robot = self.robot_fc3(robot)
        if  human.numel():
            human = self.human_fc1(human)
            human = self.activate_func(human)
            human = self.human_fc2(human)
            human = self.activate_func(human)
            human = self.human_fc3(human)

        return robot, human 
    
    def human_network(self,x):
        x, self.human_rnn_hidden = self.human_GRU(x, self.human_rnn_hidden)
        h = torch.cat(self.human_rnn_hidden.split(1), dim=-1)
        h = self.activate_func(h)
        h = self.human_fc4(h)
        return h

    def gat_network(self,data, robot_features):
        robot = None
        human = None
        robot_human = torch.zeros(len(data.ptr)-1, 32+4)

        x, edge_index = data.x, data.edge_index
        x= self.gat_conv1(x, edge_index)
        x = self.activate_func(x)
        x = self.gat_conv2(x, edge_index)

        for i in range(len(data.ptr)-1):
            origin_robot = robot_features[i,:].unsqueeze(0)
            robot = torch.index_select(x, 0, data.ptr[i])          # GAT后robot特征
            index = torch.arange(data.ptr[i]+1, data.ptr[i+1], 1)  
            if index.numel():
                human = torch.index_select(x, 0, index)            # GAT后human特征
                self.human_rnn_hidden = None
                human = self.human_network(human)                   #聚合后human特征
                robot_human[i,:] = torch.cat((robot,origin_robot ,human), dim=1) 
            else:
                pass
                a = torch.zeros(1,16)
                robot_human[i,:] = torch.cat((robot,origin_robot, a), dim=1) 
        # 变换维度
        dim = robot_human.shape
        if dim[0] > 2:
            robot_human = robot_human.reshape(2,-1,32+4)
        return robot_human
    
    
    def actor(self, x):

        x, self.actor_rnn_hidden= self.actor_rnn(x, self.actor_rnn_hidden)
        x = self.activate_func(x)
        x = self.actor_fc1(x)
        x = self.activate_func(x)
        x = self.actor_fc2(x)
        x = self.activate_func(x)
        x = self.actor_fc3(x)
        x = self.activate_func(x)
        logit = self.actor_fc4(x)           # 有点问题
        
        print(f"=============   {logit.shape}")

        return logit
    
    def critic(self, x):
        x, self.critic_rnn_hidden= self.actor_rnn(x, self.critic_rnn_hidden)
        x = self.activate_func(x)
        x = self.critic_fc1(x)
        x = self.activate_func(x)
        x = self.critic_fc2(x)
        x = self.activate_func(x)
        x = self.critic_fc3(x)
        x = self.activate_func(x)
        value = self.critic_fc4(x)

        print(f"=============   {value.shape}")

        return value

class PPO_discrete_RNN:
    def __init__(self, args):
        self.batch_size = args.batch_size               # 默认 16
        self.mini_batch_size = args.mini_batch_size     # 默认 2
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # Learning rate of actor
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.action_dim = args.action_dim

        # self.ac = Actor_Critic_RNN(args)   
        self.ac = Actor_Critic_GAT_RNN(arg) 
        # self.fea_embed = Feature_Embeding()
        # 优化算法
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)       

    def reset_rnn_hidden(self):
        self.ac.actor_rnn_hidden = None
        self.ac.critic_rnn_hidden = None

    def choose_action(self, s, evaluate=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float)
            batch_graph, robot_features = self.batch_graph_data(s)
            logit = self.ac.actor(batch_graph, robot_features)
            if evaluate:
                a = torch.argmax(logit)
                return a.item(), None
            else:
                dist = Categorical(logits=logit)
                a = dist.sample()
                a_logprob = dist.log_prob(a)
                return a.item(), a_logprob.item()

    def get_value(self, s):
        with torch.no_grad():
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
            batch_graph, robot_features = self.batch_graph_data(s)
            value = self.ac.critic(batch_graph, robot_features)
            return value.item()

    def train(self, replay_buffer, total_steps, writer):
        batch = replay_buffer.get_training_data()  # 得到训练数据:dytype: 列表
        actor_log_loss = []
        critic_log_loss = []

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # 批量采样：mini_batch_size
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # 初始化隐藏层
                self.reset_rnn_hidden()
                # print(batch['s'][index])
                batch_graph, robot_features = self.batch_graph_data(batch['s'][index])
                # print(batch_graph)
                logits_now = self.ac.actor(batch_graph, robot_features).reshape(self.mini_batch_size, -1, self.action_dim) # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
                values_now = self.ac.critic(batch_graph, robot_features).reshape(self.mini_batch_size, -1, 1).squeeze(-1)  # values_now.shape=(mini_batch_size, max_episode_len)

                dist_now = Categorical(logits=logits_now)
                dist_entropy = dist_now.entropy()  # shape(mini_batch_size, max_episode_len)
                a_logprob_now = dist_now.log_prob(batch['a'][index])  # shape(mini_batch_size, max_episode_len)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - batch['a_logprob'][index])  # shape(mini_batch_size, max_episode_len)

                # actor loss
                surr1 = ratios * batch['adv'][index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * batch['adv'][index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size, max_episode_len)
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                # critic_loss
                critic_loss = (values_now - batch['v_target'][index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                actor_log_loss.append(actor_loss)
                critic_log_loss.append(actor_loss)
                # Update
                self.optimizer.zero_grad()
                loss = actor_loss + critic_loss * 0.5
                loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

            writer.add_scalar('actor loss', sum(actor_log_loss)/len(actor_log_loss), global_step=total_steps)
            writer.add_scalar('critic loss', sum(critic_log_loss)/len(critic_log_loss), global_step=total_steps)
            
        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

  
    def save(self, checkpoint_path, total_steps, evaluate_num):
        torch.save({
            "actor_critic_model":self.ac.state_dict(),
            "total_steps":total_steps,
            "evaluate_num":evaluate_num
        }, checkpoint_path)  

    def states_to_graph(self,s):
        # 获取robot 的节点特征
        robot_features = s[0:4]
        human_features = None
        # 获取agent的个数
        num_agent = int(s[4])
        # 视野范围内有行人
        if num_agent:  
            edge_index = np.vstack((np.append(np.zeros(num_agent), np.linspace(1 ,num_agent, num_agent)),
                                    np.append(np.linspace(1 ,num_agent, num_agent),np.zeros(num_agent))),
                            )
            # 行人特征
            human_features = s[5:5+7]
            for i in range(5+7, num_agent*7, 7):
                human_features = torch.vstack((human_features, s[i:i+7]))
        # 视野范围内无行人
        else:
            edge_index = np.array([[0],[0]])
            human_features = torch.tensor([], dtype=torch.float32)
        # 特征嵌入
        robot_features = robot_features.unsqueeze(0)     # 扩充维度
        if num_agent == 1:
            human_features = human_features.unsqueeze(0)

        robot, human = self.ac.feature_embeding(robot_features, human_features)

        # return robot, human
        # 构建图
        edge_index = torch.tensor(edge_index, dtype=torch.long) # 节点
        if human.numel():
            node_feature = torch.vstack((robot, human))
        else:
            node_feature = robot
        data = Data(x=node_feature, edge_index=edge_index)     # 一张图的数据结构
        
        return data, s[0:4]

    def batch_graph_data(self,s):
        dim = s.shape
        # 变换维度
        s = s.reshape(-1, dim[2])
        # 存储批量图数据
        data_list = []
        robot_features_list = torch.zeros((s.shape[0], 4))
        for i in range(s.shape[0]):
            data, robot_features = self.states_to_graph(s[i])
            data_list.append(data)
            robot_features_list[i:] = robot_features 
        
        loader = DataLoader(dataset=data_list, batch_size=len(data_list), shuffle=False)
        batch = next(iter(loader))
        x = self.ac.gat_network(batch, robot_features_list)
        x2 = self.ac.actor(x)
        x1 = self.ac.critic(x)
        return batch, robot_features_list

    def load(self, args, checkpoint_path):
        checkpoint=torch.load(checkpoint_path)
        self.ac.load_state_dict(checkpoint["actor_critic_model"])
        total_steps = checkpoint["total_steps"]
        evaluate_num = checkpoint["evaluate_num"]
        if args.retrain:
            self.ac.train()
            print("retrain...")
        else:
            self.ac.eval()
            total_steps = 0
            evaluate_num = 0
            print("load model...")
        return total_steps, evaluate_num