from PIL import Image
from config import *

dir = "data/"

for i in range(dataset_size):
    with Image.open(dir + "occlusion_elements_{}.png".format(i)) as img:
        d = img.getdata()
        pixel_counts = {}
        for pixel in d:
            if pixel[3] > 0.0:
                pixel = pixel[:3] # remove alpha channel
                if pixel in pixel_counts:
                    pixel_counts[pixel] = pixel_counts[pixel] + 1
                else:
                    pixel_counts[pixel] = 1
        with open(dir + "voxel_scene_{}".format(i)) as scene:
            with open(dir + "visible_elements_{}".format(i), "w+") as visible:
                lines = scene.readlines()
                for line in lines:
                    data = line.split("/")
                    color = tuple([int(v) for v in data[1][1:-1].split(",")])
                    height = int(data[2]) 
                    if color in pixel_counts:
                        visible.write(str(data[0]) + "," + str(height) + "," + str(pixel_counts[color]) + "\n")