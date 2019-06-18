import numpy as np
import time

def run(L, N, dim):

    t = time.time()

    # Generate random positions of particles
    r = (np.random.random(size=(N,dim))-0.5)*L

    # uti is a list of two (1-D) numpy arrays
    # containing the indices of the upper triangular matrix
    uti = np.triu_indices(N,k=1)        # k=1 eliminates diagonal indices

    #print(uti[0])
    #print(uti[1])

    # uti[0] is i, and uti[1] is j from the previous example
    dr = r[uti[0]] - r[uti[1]]            # computes differences between particle positions
    D = np.sqrt(np.sum(dr*dr, axis=1))    # computes distances; D is a 4950 x 1 np array

    #print(dr)
    #print(D)

    print("L = " + str(L) + " N = " + str(N) + " dim = " + str(dim) + " time = " + str(time.time()-t))
    return time.time()-t

L   = 100       # simulation box dimension
dim = 3         # Dimensions

Ns = [10,100,250,500,750,1000,2000,3000,4000]
times = []
for N in Ns:
    t = run(L,N,dim)
    times.append(t)

import matplotlib.pyplot as plt
plt.plot(Ns,times)
plt.show()
