"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)
        # INSERT CODE HERE

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将输入展平为一维
        x = self.fc(x)  # 通过全连接层
        return F.log_softmax(x, dim=1)   # CHANGE CODE HERE

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        # INSERT CODE HERE

    def forward(self, x):
        x = x.view(-1 , 28*28)
        x = torch.tanh(self.fc1(x))
        x=self.fc2(x)
        return F.log_softmax(x, dim=1) # CHANGE CODE HERE

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Second convolutional layer
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 10)  # Output layer
        # INSERT CODE HERE

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply ReLU activation after the first convolutional layer
        x = F.max_pool2d(x, 2, 2)  # Apply max pooling with a 2x2 kernel
        x = F.relu(self.conv2(x))  # Apply ReLU activation after the second convolutional layer
        x = F.max_pool2d(x, 2, 2)  # Apply max pooling with a 2x2 kernel
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the fully connected layer
        x = self.fc2(x)  # Pass through the output layer
        return F.log_softmax(x, dim=1)  # CHANGE CODE HERE
