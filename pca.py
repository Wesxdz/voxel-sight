import faiss
import numpy as np

# random training data 
mt = np.random.rand(1, 4).astype('float32')
mat = faiss.PCAMatrix (4, 2)
mat.train(mt)
assert mat.is_trained
tr = mat.apply(mt)
# print this to show that the magnitude of tr's columns is decreasing
print(tr.shape)