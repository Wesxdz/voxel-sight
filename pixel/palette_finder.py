# Input a pixel image, quantize it to a count, match the quantization codes to nearest neighbors in a palette
# Apply multiple palettes simutaneously and see what looks good...

from skimage import io
import os
import sys
import argparse
import faiss
from einops import rearrange
import numpy as np

parser = argparse.ArgumentParser(description='Palette pixel art.')
parser.add_argument('--dir', type=str, default=".",
                    help='directory of pixel art')
args = parser.parse_args()
os.chdir(args.dir)

# There is probably a faster way to do this like checking the bits...
def lowest_power_of_two(n):
    i = 0
    while n > 2**i:
        i += 1
    return i

palettes = []
# alias_offsets = [(0.9, 0.9, 0.9)]
palette_dir = "/home/aeri/il/minerl/voxel_sight/palettes/"
palette_names = [file.split('.')[0] for file in os.listdir(palette_dir)]
for file in os.listdir(palette_dir):
    with open(os.path.join(palette_dir,file)) as f:
        ps = [color.strip() for color in f.readlines()]
        palette = [np.asarray([int(s[:2], base=16), int(s[2:4], base=16), int(s[4:6], base=16)]) for s in ps]
        # for offset in alias_offsets:
        #     palette.extend([np.asarray([min(255, max(0, int(c[color_idx] * offset[color_idx]))) for color_idx in range(3)]) for c in palette])
        palette.extend([np.asarray([(palette[iter_idx][color_idx] * 0.5) + (palette[iter_idx+1][color_idx] * 0.5) for color_idx in range(3)]) for iter_idx in range(len(palette) - 1)])
        
        # full_interps = []
        # for fq in range(len(palette)):
        #     full_interps.append([np.asarray([(palette[fq][color_idx] * 0.5) + (palette[iter_idx][color_idx] * 0.5) for color_idx in range(3)]) for iter_idx in range(fq, len(palette))])
        # # print(palette)
        # for interp in full_interps:
        #     palette.extend(interp)
        
        # print(palette)
        palettes.append(palette)

for i, img_path in enumerate(os.listdir(args.dir)):
    img = io.imread(img_path)
    img_h = img.shape[0]
    img_w = img.shape[1]
    xt = rearrange(img, 'h w c -> (h w) c').astype('float32')
    for j, palette in enumerate(palettes):
        # print(str(len(palette)) + " vs " + str(2**lowest_power_of_two(len(palette))))
        pq = faiss.ProductQuantizer(3, 1, lowest_power_of_two(len(palette)))
        pq.train(xt)
        codes = pq.compute_codes(xt)
        # TODO: Map codes onto palette colors!
        x2 = pq.decode(codes)
        code_to_palette_index = {}
        for code_idx, code in enumerate(codes):
            if int(code) in code_to_palette_index:
                pass
            else:
                dc = x2[code_idx]
                # For each decoded color code, get the closest palette color
                # Just brute force since we try to get to MVP fastest to validate hypothesis
                dists = []
                for pal_idx, color in enumerate(palette):
                    dist = sum([(dc[color_idx] - color[color_idx])**2 for color_idx in range(3)])
                    dists.append((dist, pal_idx))
                dists = sorted(dists)
                code_to_palette_index[int(code)] = dists[0][1]

        # TODO: Save an image with palette
        img_with_palette = [palette[code_to_palette_index[int(code)]] for code in codes]
        approximate = rearrange(img_with_palette, '(h w) c -> h w c', h=img_h, w=img_w)
        print(approximate)
        io.imsave(palette_names[j] + ".png", approximate)
