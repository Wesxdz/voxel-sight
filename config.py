downsample = 4
# C x H x W
screen_size = (4, int(360/downsample), int(640/downsample))
quantized_screen_size = (screen_size[1] * screen_size[2], 2)

voxel_grid_size = (16*16,)
dataset_size = 1000