from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import faiss
from einops import rearrange
from config import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class VoxelViewDataset(Dataset):
    """
    Monocular forward facing views of 64x64 voxel grid
    45, 115 x rotation
    -44 to 44 z rotation
    """
    
    def __init__(self, dataset_size, dir, transform=None):
        self.dataset_size = dataset_size
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return self.dataset_size

    def get_camera_view(self, idx):
        img = io.imread(os.path.join(self.dir, "voxels_{}.png".format(idx)))
        d = 4
        cs = 1
        colors = rearrange(img, 'w h c -> (w h) c')
        print(colors.shape)
        pq = faiss.ScalarQuantizer(d, cs)
        pq.train(colors)
        # codes = pq.compute_codes(colors)
        # print(codes)
        return torch.from_numpy(img)

    def get_voxel_grid(self, idx):
        height_grid = torch.zeros(voxel_grid_size, dtype=torch.int8)
        reward_grid = torch.zeros(voxel_grid_size, dtype=torch.int32)
        with open(os.path.join(self.dir, "visible_elements_{}".format(idx))) as visible:
            lines = visible.readlines()
            for line in lines:
                data = [int(d) for d in line.split(",")]
                height_grid[data[0]] = data[1]
                reward_grid[data[0]] = data[2]
        return height_grid, reward_grid


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        camera_view = self.get_camera_view(idx)
        # TODO: Use Faiss to generate k means so that the input data can use a color index byte to reduce input Tensor size while retaining color
        height_grid, reward_grid = self.get_voxel_grid(idx)

        sample = {'view':camera_view, 'grid':height_grid, 'rewards':reward_grid}

        if self.transform:
            sample = self.transform(sample)

        return sample