from ast import arg
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

# GAT 层
class GATLayer(nn.Module):
    """
    Simple PyTorch Implementation of the Graph Attention layer.
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout       = dropout        # drop prob = 0.6
        self.in_features   = in_features    # 
        self.out_features  = out_features   # 
        self.alpha         = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat        = concat         # conacat = True for all layers except the output layer.

        # Xavier Initialization of Weights
        # W , a 学习参数
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # 激活函数：LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        # 线性变换
        h = torch.mm(input, self.W)
        N = h.size()[0]

        # Attention Mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e       = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec  = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime   = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

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

        # actor
        # GAT
        self.actor_conv1 = GATConv(self.input_features, self.hid, heads=self.in_head, dropout=0.6)
        self.actor_conv2 = GATConv(self.hid*self.in_head, self.output_features, concat=False,
                             heads=self.out_head, dropout=0.6)
        # GRU
        self.actor_fc1 = nn.Linear(self.output_features + 4,self.fc1_hidden)
        self.actor_rnn = nn.GRU(self.fc1_hidden, self.rnn_hidden_dim, batch_first=True)
        self.actor_fc2 = nn.Linear(self.rnn_hidden_dim, self.action_dim)

        # critic
        # GAT
        self.critic_conv1 = GATConv(self.input_features, self.hid, heads=self.in_head, dropout=0.6)
        self.critic_conv2 = GATConv(self.hid*self.in_head, self.output_features, concat=False,
                             heads=self.out_head, dropout=0.6)
        # GRU
        self.critic_fc1 = nn.Linear(self.output_features + 4,self.fc1_hidden)
        self.critic_rnn = nn.GRU(self.fc1_hidden, self.rnn_hidden_dim, batch_first=True)
        self.critic_fc2 = nn.Linear(self.rnn_hidden_dim, 1)

    def actor(self, data, robot_features):
        x, edge_index = data.x, data.edge_index
        # robot 节点索引
        robot_index = data.ptr[:-1]
        x = self.actor_conv1(x, edge_index) 
        x = self.activate_func(x)
        x = self.actor_conv2(x, edge_index)
        # 提取 GAT 后 robot 的特征
        x = torch.index_select(x, 0, robot_index)
        # 拼接robot的特征
        x = torch.cat((x, robot_features),dim=1)
        # GRU
        x = self.actor_fc1(x)
        x = self.activate_func(x)
        x, self.actor_rnn_hidden = self.actor_rnn(x, self.actor_rnn_hidden)
        logit = self.actor_fc2(x)
        return logit
    def critic(self, data, robot_features):
        x, edge_index = data.x, data.edge_index
        # robot 节点索引
        robot_index = data.ptr[:-1]
        x = self.critic_conv1(x, edge_index)
        x = self.activate_func(x)
        x = self.critic_conv2(x, edge_index)
        # 提取 GAT 后 robot 的特征
        x = torch.index_select(x, 0, robot_index)
        # 拼接robot的特征
        x = torch.cat((x, robot_features),dim=1)
        # GRU
        x = self.critic_fc1(x)
        x = self.activate_func(x)
        x, self.critic_rnn_hidden = self.critic_rnn(x, self.critic_rnn_hidden)
        value = self.critic_fc2(x)
        return value

# class Actor_Critic_RNN(nn.Module):
#     def __init__(self, args):
#         super(Actor_Critic_RNN, self).__init__()
#         self.use_gru = args.use_gru
#         self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

#         self.actor_rnn_hidden = None
#         self.actor_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
#         if args.use_gru:
#             print("------use GRU------")
#             self.actor_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
#         else:
#             print("------use LSTM------")
#             self.actor_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
#         self.actor_fc2 = nn.Linear(args.hidden_dim, args.action_dim)

#         self.critic_rnn_hidden = None
#         self.critic_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
#         if args.use_gru:
#             self.critic_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
#         else:
#             self.critic_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
#         self.critic_fc2 = nn.Linear(args.hidden_dim, 1)

#         if args.use_orthogonal_init:
#             print("------use orthogonal init------")
#             orthogonal_init(self.actor_fc1)
#             orthogonal_init(self.actor_rnn)
#             orthogonal_init(self.actor_fc2, gain=0.01)
#             orthogonal_init(self.critic_fc1)
#             orthogonal_init(self.critic_rnn)
#             orthogonal_init(self.critic_fc2)

#     def actor(self, s):
#         s = self.activate_func(self.actor_fc1(s))
#         output, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden)
#         logit = self.actor_fc2(output)
#         return logit

#     def critic(self, s):
#         s = self.activate_func(self.critic_fc1(s))
#         output, self.critic_rnn_hidden = self.critic_rnn(s, self.critic_rnn_hidden)
#         value = self.critic_fc2(output)
#         return value


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
                logits_now = self.ac.actor(batch_graph, robot_features).reshape(self.mini_batch_size, -1, self.action_dim)              # logits_now.shape=(mini_batch_size, max_episode_len, action_dim)
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
        # 获取agent的个数
        num_agent = int(s[4])
        if num_agent == 0:
            num_agent = 3 
        # 边索引
        a = np.zeros(num_agent)
        b = np.linspace(1 ,num_agent, num_agent)
        a = np.append(a, b)
        edge_index = np.vstack((np.append(np.zeros(num_agent), np.linspace(1 ,num_agent, num_agent)),
                                    np.append(np.linspace(1 ,num_agent, num_agent),np.zeros(num_agent))),
                            )
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # 节点特征 
        node_feature = s[0:4]
        for i in range(5, num_agent*7, 7):
            node_feature = np.vstack((node_feature, s[i:i+4]))
        node_feature = torch.tensor(node_feature, dtype=torch.float)
        # 一张图的数据结构
        data = Data(x=node_feature, edge_index=edge_index)
        
        return data, robot_features

    def batch_graph_data(self,s):
        # 变换维度
        s = s.reshape(-1, 68)
        # 存储批量图数据
        data_list = []
        robot_features_list = torch.zeros((s.shape[0], 4))
        for i in range(s.shape[0]):
            data, robot_features = self.states_to_graph(s[i])
            data_list.append(data)
            robot_features_list[i:] = robot_features 
        
        loader = DataLoader(dataset=data_list, batch_size=len(data_list), shuffle=False)
        batch = next(iter(loader))
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