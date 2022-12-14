downsample = 4
native_screen_res = (640, 360)
# C x H x W
screen_size = (4, int(native_screen_res[1]/downsample), int(native_screen_res[0]/downsample))
quantized_screen_size = (screen_size[1] * screen_size[2], 2)

voxel_grid_size = (16*16,)

datasets = {'train':100}
# datasets = {'train':5000, 'test':1000, 'validate':500}
