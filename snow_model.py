import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):

    def __init__(self, hidden=1024):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(7, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, hidden)
        self.fc6 = nn.Linear(hidden, hidden)
        self.fc7 = nn.Linear(hidden, 1)
        self.register_buffer('norm',
                             torch.tensor([10.0,
                                           8.5,
                                           10.4,
                                           0.025,
                                           0.35,
                                           0.20,
                                           0.025]))

    def forward(self, x):
        # normalize the parameter to range [0-1] 
        x = x / self.norm
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = F.elu(self.fc5(x))
        x = F.elu(self.fc6(x))
        return self.fc7(x)
