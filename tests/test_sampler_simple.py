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
    sampler.rejection_sample_first_particle(no_tries_max)
    sampler.rejection_sample_second_particle(no_tries_max)

    sampler.cdf_sample_first_particle()
    sampler.cdf_sample_second_particle()
