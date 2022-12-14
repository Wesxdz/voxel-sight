from audioop import avg
from pickletools import uint4
from turtle import forward
from typing import Sequence
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
# from halonet_pytorch import HaloAttention
from voxel_dataset import VoxelHeightDataset, VoxelVisibilityDataset
from config import *
import numpy as np
from einops import rearrange, reduce, repeat
import os
from PIL import Image
import math
# import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"

# wandb.init(project="voxel_sight")

# Hyperparameters
num_epochs = 80
learning_rate = 0.001

train = False

visibility_training_dataset = None
visibility_training_dataloader = None

height_training_dataset = None
height_training_dataloader = None

# Train two neural networks, one for visibility, another for height

if train:
    visibility_training_dataset = VoxelVisibilityDataset(datasets['train'], os.path.join("data", "train"))
    visibility_training_dataloader = DataLoader(visibility_training_dataset, batch_size=4,
                            shuffle=True, num_workers=8)
    height_training_dataset = VoxelHeightDataset(datasets['train'], os.path.join("data", "train"))
    height_training_dataloader = DataLoader(height_training_dataset, batch_size=4,
                            shuffle=True, num_workers=8)
else:
    visibility_training_dataset = VoxelVisibilityDataset(datasets['train'], os.path.join("data", "train"))
    visibility_training_dataloader = DataLoader(visibility_training_dataset, batch_size=4,
                            shuffle=False, num_workers=1)
    height_training_dataset = VoxelHeightDataset(datasets['train'], os.path.join("data", "train"))
    height_training_dataloader = DataLoader(height_training_dataset, batch_size=4,
                            shuffle=False, num_workers=1)

# test_dataset = VoxelViewDataset(datasets['test'], os.path.join("data", "test"))
# test_dataloader = DataLoader(test_dataset, batch_size=16,
#                         shuffle=True, num_workers=8)
                        
# test_data = None # TODO: Create a test dataset!

# HaloNet implementation 
# https://arxiv.org/pdf/2103.12731.pdf
# based on ResNet architecture
# https://arxiv.org/pdf/1512.03385.pdf
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/deep_residual_network/main.py

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

view_channels = np.prod(quantized_screen_size)
print(view_channels)
voxel_channels = np.prod(voxel_grid_size)
print(voxel_channels)

class SimpleTest(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(SimpleTest, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, dtype=torch.half)

    def forward(self, x):
        out = self.linear(x)
        return out

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
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 16
        # TODO: How to have variable downsampling?
        self.conv = conv3x3(3, 16) #TODO Quantize here!
        # self.halo = HaloAttention(dim=screen_size[1]*screen_size[2], block_size=10, halo_size=4, dim_head=64, heads=4)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 16*2, layers[1], 2)
        self.layer3 = self.make_layer(block, 16*4, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.occlusion = nn.Linear(640, 640)
        self.fc = nn.Linear(640, voxel_channels)
        # self.linear = nn.Linear(3600, voxel_channels, dtype=torch.half)
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        # x = rearrange(x, 'b c w h -> b (c w h)')
        # out = self.halo(x)
        # print(out.shape)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.occlusion(out)
        out = self.fc(out)
        return out

def get_reward(actual, prediction, rewards):
    reward = 0
    for i in len(actual):
        if actual[i] == prediction[i]:
            reward += rewards[i]
    return reward

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def voxelLoss(outputs, labels):
    pass #TODO integer loss on only visible voxels?

# https://arxiv.org/pdf/1308.3432.pdf
# Straight-Through Estimator for thresholding hidden voxels?

if train:
    nns = [visibility_training_dataloader, height_training_dataloader]
    nn_names = ["visible_voxels", "height_voxels"]
    for ni, nndata in enumerate(nns):
        model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(nndata)
        curr_lr = learning_rate

        for epoch in range(num_epochs):
            print("Start epoch {}".format(epoch))
            for i, sample in enumerate(nndata):
                view = sample['view']
                view = view.to(device)
                grid = sample['grid'].to(device)
                outputs = model(view)
                loss = criterion(outputs, grid)
                # wandb.log({"loss":loss})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 25 == 0:
                    print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
            # Decay learning rate
            if (epoch+1) % 20 == 0:
                curr_lr /= 3
                update_lr(optimizer, curr_lr)

        torch.save(model.state_dict(), '{}.ckpt'.format(nn_names[ni]))
else:
    visible_model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
    height_model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
    visible_model.load_state_dict(torch.load("visible_voxels.ckpt"))
    visible_model.eval()
    height_model.load_state_dict(torch.load("height_voxels.ckpt"))
    height_model.eval()
    with torch.no_grad():
        i = 0
        for sample in visibility_training_dataloader:
            view = sample['view']
            view = view.to(device)
            grid = sample['grid'].to(device)
            visiblility_outputs = visible_model(view)
            height_outputs = height_model(view)
            for b, x in enumerate(sample['grid']):
                actual_img = Image.new('RGB', (16, 16))
                actual = rearrange(x, '(b1 b2) -> b1 b2', b1=16, b2=16)
                predicted_visibility = rearrange(visiblility_outputs[b], '(b1 b2) -> b1 b2', b1=16, b2=16)
                predicted_height = rearrange(height_outputs[b], '(b1 b2) -> b1 b2', b1=16, b2=16)

                compare = Image.new('RGB', (64 + 5, 16 + 2))

                visible_elements = "./data/train/visible_elements_{}".format(i)
                with open(visible_elements) as visible:
                    voxels = [line.strip() for line in visible.readlines()]
                    for v in [voxel for voxel in voxels]:
                        data = v.split(',')
                        index = int(data[0])
                        height = int(data[1])

                        
                        visibile_pixels = int(data[2])
                        coord = (int(index) % 16, int(index/16))
                        brightness = max(0.2, min(int(math.pow(visibile_pixels, .95)), 256)/64)
                        red = 256 + height * 32
                        
                        green = 128 + height * 32
                        blue = 128 + height * 32

                        if height > 0:
                            red = 0

                        elif height < 0:
                            blue = 0
                            green = 0
                        else:
                            red = 128
                        actual_img.putpixel(coord, (int(red * brightness), int(green * brightness), int(blue * brightness)))
                        # predited_img.putpixel()
                actual_img = actual_img.transpose(Image.FLIP_TOP_BOTTOM)
                
                compare.paste(actual_img, (1, 1))

                # img = (predicted.cpu().clone() * 32 + 128).clamp(0, 255).numpy()
                visibility_img = (predicted_visibility.cpu().clone() ).clamp(0, 255).numpy()
                predicted_visibility_img = Image.fromarray(visibility_img)
                predicted_visibility_img = predicted_visibility_img.transpose(Image.FLIP_TOP_BOTTOM)
                compare.paste(predicted_visibility_img, (16 + 2, 1))

                height_img = (predicted_height.cpu().clone() * 32 + 128).clamp(0, 255).numpy()
                predicted_height_img = Image.fromarray(height_img)
                predicted_height_img = predicted_height_img.transpose(Image.FLIP_TOP_BOTTOM)
                compare.paste(predicted_height_img, (32 + 3, 1))

                world_prediction = Image.new('RGB', (16, 16))
                for index in range(256):
                    coord = (int(index) % 16, int(index/16))
                    predicted_visible_pixel = visiblility_outputs[b][index].item()
                    # print(predicted_visible_pixel)
                    predicted_height = round(height_outputs[b][index].item())
                    brightness = max(0.2, min(int(math.pow(max(0, predicted_visible_pixel), .95)), 256)/64)
                    red = 256 + predicted_height * 32
                    
                    green = 128 + predicted_height * 32
                    blue = 128 + predicted_height * 32

                    if predicted_height > 0:
                        red = 0

                    elif predicted_height < 0:
                        blue = 0
                        green = 0
                    else:
                        red = 128
                    if predicted_visible_pixel > 0.0:
                        world_prediction.putpixel(coord,(int(red * brightness), int(green * brightness), int(blue * brightness)))

                world_prediction = world_prediction.transpose(Image.FLIP_TOP_BOTTOM)
                compare.paste(world_prediction, (48 + 4, 1))
                

                compare = compare.resize((compare.size[0]*2, compare.size[1]*2), Image.NEAREST)
                compare.save('compare_{}.png'.format(i))
                i += 1
                # for i, v in enumerate(x):
                #     if v != 0.0:
                #         print('{} vs {}'.format(v, outputs[b][i]))
            # print(outputs)
            # print(sample['grid'])