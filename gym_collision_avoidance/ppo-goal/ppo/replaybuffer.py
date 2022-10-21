import torch
import numpy as np

# 存储序列
class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, args.state_dim))    # 状态
        self.a = np.zeros((args.batch_size, 1))                 # 动作
        self.a_logprob = np.zeros((args.batch_size, 1))         # 该动作概率的log()
        self.r = np.zeros((args.batch_size, 1))                 # 奖励
        self.s_ = np.zeros((args.batch_size, args.state_dim))   # 下一转状态
        self.dw = np.zeros((args.batch_size, 1))                # 是否超时
        self.done = np.zeros((args.batch_size, 1))              # 是否结束一个episode
        self.count = 0                                          # 计数

    def store(self, s, a, a_logprob, r, s_, dw, done):          # 存储序列
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):                  # 数据类型转换
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done
