from scipy.spatial import Delaunay
import numpy as np

import random

n = 30
points = []
for i in range(0,n):
    points.append([random.uniform(-5.0,5.0),random.uniform(-5.0,5.0),random.uniform(-5.0,5.0)])
points = np.array(points)

tri = Delaunay(points)

p = np.array([random.uniform(-5.0,5.0),random.uniform(-5.0,5.0),random.uniform(-5.0,5.0)])
f = tri.find_simplex(p)
print(f)
print(tri.simplices[f])
