import sys
import os
import random

# TODO: Parse through palettes
palettes = []
for file in os.listdir("palettes"):
    with open("palettes/" + file) as p:
        palettes.append(["#" +  line.rstrip() for line in p.readlines()])
random_palette = random.choice(palettes)
# probably need to select a random subset of this per tile, otherwise it's kind of noisy

terrain_solid = [
    "grass",
    "stone",
    "clay",
    "coal ore",
    "brick",
    "wooden planks",
    "asphalt",
    "sand",
    "plastic",
    "plaster",
    "wool",
    "leather",
    "carbon",
    "metal",
    "ceramic",
    "paper",
    "cloth",
    "dirt",
    "gravel",
    "shells",
    "pebbles",
    "sandstone",
    "redrock",
    "mud",
    "steel",
    "cement render",
    "quarry tile",
    "mosaic",
    "terrazzo",
    "carpet",
    "marble",
    "wallpaper",
    "paint",
    "moss"
]

terrain_top = [

]

os.chdir("pixray")
origin_dir = os.path.abspath(os.curdir)
# --outdir="' + origin_dir + '/grass"
command = 'sudo cog run python pixray.py --drawer=pixel --prompt="flat, top down view, grass texture #pixelart by aamatniekss, Mrmo Tarius, and slym" --size 64 64 --palette=['
for color in random_palette:
    command += str(color) + ","
command = command[:-1]
command += "]"
print(command)
os.system(command)