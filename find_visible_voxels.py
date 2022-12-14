from PIL import Image
from config import *
import os

for dataset in datasets.items():
    dir = os.path.join("data", dataset[0])
    for i in range(dataset[1]):
        with Image.open(os.path.join(dir, "occlusion_elements_{}.png".format(i))) as img:
            d = img.getdata()
            pixel_counts = {}
            for pixel in d:
                if pixel[3] > 0.0:
                    pixel = pixel[:3] # remove alpha channel
                    if pixel in pixel_counts:
                        pixel_counts[pixel] = pixel_counts[pixel] + 1
                    else:
                        pixel_counts[pixel] = 1
            with open(os.path.join(dir, "voxel_scene_{}".format(i))) as scene:
                with open(os.path.join(dir, "visible_elements_{}".format(i)), "w+") as visible:
                    lines = scene.readlines()
                    for line in lines:
                        data = line.split("/")
                        color = tuple([int(v) for v in data[1][1:-1].split(",")])
                        height = int(data[2]) 
                        if color in pixel_counts:
                            visible.write(str(data[0]) + "," + str(height) + "," + str(pixel_counts[color]) + "\n")