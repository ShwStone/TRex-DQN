import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DinoDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DinoDQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[1], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        
        # Dueling DQN
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape[1:]))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))
