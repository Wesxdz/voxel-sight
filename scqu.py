import faiss
import numpy as np
from skimage import io, transform
import os
from einops import rearrange
import sys

d = 4  # data dimension
dataset_size = 32

# train set
img = io.imread(os.path.join("data", "voxels_0.png"))
xt = rearrange(img, 'h w c -> (h w) c').astype('float32')
print(sys.getsizeof(xt))

# QT_8bit allocates 8 bits per dimension (QT_4bit also works)
sq = faiss.ScalarQuantizer(d, faiss.ScalarQuantizer.QT_4bit)
sq.train(xt)

# encode 
codes = sq.compute_codes(xt)
print(sys.getsizeof(codes))
print(type(codes[0][1]))

# decode
x2 = sq.decode(codes)

comp = rearrange(x2, '(h w) c -> h w c', h=90, w=160)
io.imsave("test.png", comp)

# compute reconstruction error
# avg_relative_error = ((xt - x2)**2).sum() / (xt ** 2).sum()
# print(codes)
# print(avg_relative_error)