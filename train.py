from audioop import avg
from typing import Sequence
import torch
from torch import nn
from torch.utils.data import DataLoader
from halonet_pytorch import HaloAttention
from voxel_dataset import VoxelViewDataset
from config import *
from math import prod
# from einops import

# Hyperparameters
num_epochs = 80
batch_size = 100
learning_rate = 0.001

training_dataset = VoxelViewDataset(12, "data")
dataloader = DataLoader(training_dataset, batch_size=4,
                        shuffle=True, num_workers=3)

for i_batch, sample_batched in enumerate(dataloader):
    print(sample_batched.keys())

# test_data = None # TODO: Create a test dataset!

# TODO: Train from input to output tensor with nn architecture!
device = "cuda" if torch.cuda.is_available() else "cpu"

# HaloNet implementation 
# https://arxiv.org/pdf/2103.12731.pdf
# based on ResNet architecture
# https://arxiv.org/pdf/1512.03385.pdf

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super(NeuralNetwork, self).__init__()
        view_ch = prod(screen_size)
        voxel_ch = prod(voxel_grid_size)
        self.residual = nn.Sequential(
            nn.Conv2d(view_ch, view_ch, (7, 7, ),
            nn.Conv2d(view_ch, view_ch, (7, 7)),
            nn.
            nn.AvgPool2d(),
            nn.Linear(view_ch, voxel_ch)
        )


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             HaloAttention(
#                 dim = screen_input_size/4,
#                 block_size = 8,
#                 halo_size = 4,
#                 dim_head = 64,
#                 heads = 4
#             ),
#             nn.ReLU(),
#             nn.Linear(screen_input_size/4, voxel_grid_size[0])
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits

# model = NeuralNetwork().to(device)
# print(model)

# print(model)


# def get_reward(actual, prediction, rewards):
#     reward = 0
#     for i in len(actual):
#         if actual[i] == prediction[i]:
#             reward += rewards[i]
#     return reward