import numpy as np

r = np.random.randn(100,3)

H, edges = np.histogramdd(r)
print(H.shape, edges[0].size, edges[1].size, edges[2].size)