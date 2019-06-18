import numpy as np
import time

def run(L, N, dim, std_dev):

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
    D_squared = np.sum(dr*dr, axis=1)    # computes distances squared; D is a 4950 x 1 np array

    t1 = time.time()-t

    #print(dr)
    #print(D)

    # Clip distances at 3 sigma
    max_dist_squared = pow(3*std_dev,2)
    two_var = 2.0 * pow(std_dev,2)

    tt = time.time()

    '''
    z = zip(uti[0],uti[1],D_squared)
    zfilter = [(idx1,idx2,dist_squared) for idx1,idx2,dist_squared in z if dist_squared < max_dist_squared]
    uti0filter, uti1filter, D_squared_filter = zip(*zfilter)
    '''
    stacked = np.array([uti[0],uti[1],D_squared]).T
    uti0filter, uti1filter, D_squared_filter = stacked[stacked[:,2] < max_dist_squared].T
    exps = np.exp(-D_squared_filter/two_var)
    probs = np.bincount(uti0filter.astype(int),exps,minlength=N) + np.bincount(uti1filter.astype(int),exps,minlength=N)
    # rs = range(0,N)
    # probs1 = np.sum(exps[np.equal.outer(uti0filter, rs)],axis=0)
    # print(probs1)
    # probs = np.array([ np.sum(exps[np.logical_or(uti0filter == idx, uti1filter == idx)]) for idx in range(0,N)])
    norm = np.sum(probs)
    probs /= norm
    print(probs)

    idx1d = 2
    probs1d = exps[np.logical_or(uti0filter == idx1d, uti1filter == idx1d)])
    norm1d = np.sum(probs1d)
    probs1d /= norm1d

    # uti0filter, uti1filter, D_squared_filter = zip(*((idx1,idx2,dist_squared) for idx1, idx2, dist_squared in zip(uti[0],uti[1],D_squared) if dist_squared < max_dist_squared))
    # exps = np.exp( - np.array(D_squared_filter) / two_var)

    tt1 = time.time()
    print("done: " + str(tt1 - tt))

    norms = []
    len_exps = len(exps)
    for iN in range(0,N):
        # norms.append(sum([exps[i] for i in range(0,len_exps) if uti0filter[i] == iN or uti1filter[i] == iN]))
        norms.append(sum(exps[[i for i in range(0,len_exps) if uti0filter[i] == iN or uti1filter[i] == iN]]))

    norm_all = sum(norms)
    probs = [norm / norm_all for norm in norms]
    print(probs)

    tt2 = time.time()
    print(tt2 - tt1)

    exps = {}
    idxs_other = {}
    for i in range(0,N):
        exps[i] = []
        idxs_other[i] = []
    for i in range(0,len(D_squared)):
        dist_squared = D_squared[i]
        if dist_squared < max_dist_squared:
            exp_term = np.exp( - dist_squared / two_var)
            idx1 = uti[0][i]
            idx2 = uti[1][i]
            exps[idx1].append(exp_term)
            idxs_other[idx1].append(idx2)
            exps[idx2].append(exp_term)
            idxs_other[idx2].append(idx1)

    tt3 = time.time()
    print(tt3 - tt2)

    t2 = time.time()-t

    norms = {}
    for i in range(0,N):
        norms[i] = sum(exps[i])
    norm_all = sum(list(norms.values()))

    probs = {}
    for i in range(0,N):
        probs[i] = norms[i] / norm_all

    t3 = time.time()-t
    ts = [t1,t2,t3]

    print("L = " + str(L) + " N = " + str(N) + " dim = " + str(dim) + " times = " + str(ts))

    return [ts,probs]

L   = 100       # simulation box dimension
# Particles are spread -L/2 to L/2 uniformly
dim = 3         # Dimensions
std_dev = 10.0

N = 100
ts,probs = run(L,N,dim,std_dev)
'''
import matplotlib.pyplot as plt
plt.plot(probs.keys(),probs.values())
plt.show()
'''

'''
Ns = [10,100,250,500,750,1000,2000,3000,4000]
times1 = []
times2 = []
for N in Ns:
    t1,t2 = run(L,N,dim,std_dev)
    times1.append(t1)
    times2.append(t2)

import matplotlib.pyplot as plt
plt.plot(Ns,times1)
plt.plot(Ns,times2)
plt.show()
'''
