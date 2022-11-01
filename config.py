downsample = 4
# C x H x W
screen_size = (4, int(360/downsample), int(640/downsample))
# screen_size = (128, 128)
# TODO: The screen size is too large to fit in GPU memory and is probably overkill
# Let's downscale the resolution of the input images and represent the four channels as one

# screen_input_size = 4 * 360 * 640
voxel_grid_size = (16*16,)
dataset_size = 32