# If installed:
from samplePairsGaussian import *

# Else: Add the path to the module
# import sys
# sys.path.append('../samplePairsGaussian/')
# from sampler import *

import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Uniformly spread particles from -L to L
    L = 100

    # Dimensionality (1D,2D,3D, etc. - all dims are supported, but only 1D will be possible to plot pairs and probability)
    dim = 1

    # Number of particles
    N = 100

    # Positions
    posns = (np.random.random(size=(N,dim))-0.5) * (2.0 * L)

    # Setup the sampler

    # Make the probability calculator
    std_dev = 10.0
    std_dev_clip_mult = 3.0
    prob_calculator = ProbCalculator(posns,dim,std_dev,std_dev_clip_mult)
    # Distances have already been computed for us between all particles

    # Make the sampler
    sampler = Sampler(prob_calculator)

    # Sample many particle pairs

    # Handle failure
    def handle_fail():
        print("Could not draw particle: try adjusting the std. dev. for the probability cutoff.")
        sys.exit(0)

    no_samples = 1000
    no_tries_max = 100
    idxs_1 = []
    idxs_2 = []
    for i in range(0,no_samples):

        # Sample using rejection sampling
        success = sampler.rejection_sample(no_tries_max=no_tries_max)
        if not success:
            handle_fail()

        # Alternatively sample using CDF
        #success = sampler.cdf_sample_pair_given_nonzero_probs_for_first_particle()
        #if not success:
        #    handle_fail()

        # Append
        idxs_1.append(sampler.idx_first_particle)
        idxs_2.append(sampler.idx_second_particle)

    idxs_1 = np.array(idxs_1).astype(int)
    idxs_2 = np.array(idxs_2).astype(int)

    # Plot

    plt.figure()
    plt.hist(posns)
    plt.xlabel("particle position")
    plt.title("Distribution of " + str(N) + " particle positions")

    plt.figure()
    # Make symmetric
    idxs_1_symmetric = np.concatenate((idxs_1,idxs_2))
    idxs_2_symmetric = np.concatenate((idxs_2,idxs_1))
    plt.plot(posns[idxs_1_symmetric][:,0],posns[idxs_2_symmetric][:,0],'o')
    plt.xlabel("position of particle #1")
    plt.ylabel("position of particle #2")
    plt.title(str(no_samples)+ " sampled pairs of particles")

    # Show
    plt.show()
