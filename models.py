import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import set_seed
from constants import *

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1)
        self.dropout = nn.Dropout(0.2)
        self.batchnorm2d_1 = nn.BatchNorm2d(128)
        self.batchnorm2d_2 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*2*2, 128)
        self.fc2 = nn.Linear(128, 100) # 100 classes for fine labels

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.batchnorm2d_1(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv4(x))
        x = self.batchnorm2d_2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1)
        return output

def get_model(seed=None):
    if seed:
        set_seed(seed)
    model = Net()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    return model, optimizer
    

class AttackNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(100, 256) # input = shadow model probs for CIFAR-100
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 1)

  def forward(self, x):
    x = F.dropout(F.relu(self.fc1(x)))
    x = F.dropout(F.relu(self.fc2(x)))
    x = self.fc3(x)
    return F.log_softmax(x, dim=1)

def get_attack_model():
  model = AttackNet()
  optimizer = torch.optim.AdamW(model.parameters())
  return model, optimizer