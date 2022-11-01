from audioop import avg
from typing import Sequence
import torch
from torch import nn
from torch.utils.data import DataLoader
from halonet_pytorch import HaloAttention
from voxel_dataset import VoxelViewDataset
from config import *
import numpy as np
from einops import rearrange, reduce, repeat

# Hyperparameters
num_epochs = 80
batch_size = 1
learning_rate = 0.001

training_dataset = VoxelViewDataset(12, "data")
dataloader = DataLoader(training_dataset, batch_size=1,
                        shuffle=True, num_workers=1)

for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched.keys())

# test_data = None # TODO: Create a test dataset!

# TODO: Train from input to output tensor with nn architecture!
device = "cuda" if torch.cuda.is_available() else "cpu"

# HaloNet implementation 
# https://arxiv.org/pdf/2103.12731.pdf
# based on ResNet architecture
# https://arxiv.org/pdf/1512.03385.pdf

# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

view_channels = np.prod(screen_size)
voxel_channels = np.prod(voxel_grid_size)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None) -> None:
        super(ResidualBlock, self).__init__()
        # TODO Replace convolutional layers with HaloAttention
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        def forward(self, x):
            residual = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2d(out)
            out = self.bn2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out

class ResNet(nn.Module):
    def __init__(self, block, layers) -> None:
        super(ResNet, self).__init__()
        self.in_channels = view_channels
        self.conv = conv3x3(3, view_channels)
        self.bn = nn.BatchNorm2d(view_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, view_channels, layers[0])
        self.layer2 = self.make_layer(block, view_channels*2, layers[1], 2)
        self.layer3 = self.make_layer(block, view_channels*4, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(view_channels/2)
        self.fc = nn.Linear(view_channels*4, voxel_channels)
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def get_reward(actual, prediction, rewards):
    reward = 0
    for i in len(actual):
        if actual[i] == prediction[i]:
            reward += rewards[i]
    return reward

model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
total_step = len(training_dataset)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, sample in enumerate(training_dataset):
        outputs = model()
        loss = criterion(outputs, sample['grid'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# model.eval()
# with torch.no_grad():
    # for sample in test_dataset:


torch.save(model.state_dict(), 'voxelsight.ckpt')