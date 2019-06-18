# Add the path to the module
import sys
sys.path.append('../samplePairsGaussian/')
import numpy as np

from sampler import *

if __name__ == "__main__":

    # Generate random positions of particles
    L = 100
    dim = 1
    N = 100
    posns = (np.random.random(size=(N,dim))-0.5)*L

    # Sampler
    std_dev = 10.0
    std_dev_clip_mult = 3.0
    sampler = Sampler(posns,std_dev,std_dev_clip_mult)

    no_tries_max = 100
    no_samples = 100
    idxs_1 = []
    idxs_2 = []
    for i in range(0,no_samples):
        sampler.rejection_sample_first_particle(no_tries_max)
        sampler.rejection_sample_second_particle(no_tries_max)
        idxs_1.append(sampler.idx_first_particle)
        idxs_2.append(sampler.idx_second_particle)

    import matplotlib.pyplot as plt
    plt.plot(posns[idxs_1],posns[idxs_2],'o')
    plt.show()
