# Add the path to the module
import sys
sys.path.append('../samplePairsGaussian/')

import numpy as np
import matplotlib.pyplot as plt

from sampler import *

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

    # Cutoff counting probabilities for particles that are more than:
    # std_dev_clip_mult * std_dev
    # away from each-other
    std_dev = 10.0
    std_dev_clip_mult = 3.0

    # Make the probability calculator
    prob_calculator = ProbCalculator(posns,std_dev,std_dev_clip_mult)
    # Distances have already been computed for us between all particles

    # Make the sampler
    sampler = Sampler(prob_calculator)

    # Sample many particle pairs

    # Handle failure
    def handle_fail():
        print("Could not draw particle: try adjusting the std. dev. for the probability cutoff.")
        sys.exit(0)

    # For efficiency, just compute the first particle probability now
    prob_calculator.compute_probs_first_particle()
    compute_probs_first_particle = False

    no_samples = 10000
    no_tries_max = 100
    idxs_1 = []
    idxs_2 = []
    for i in range(0,no_samples):

        # Sample using rejection sampling
        success = sampler.rejection_sample_first_particle(no_tries_max,compute_probs_first_particle)
        if not success:
            handle_fail()

        success = sampler.rejection_sample_second_particle(no_tries_max)
        if not success:
            handle_fail()

        # Alternatively sample using CDF
        # sampler.cdf_sample_first_particle()
        # sampler.cdf_sample_second_particle()

        # Append
        idxs_1.append(sampler.idx_first_particle)
        idxs_2.append(sampler.idx_second_particle)

    # Plot

    # Histogram of the particle positions
    plt.figure()
    plt.hist(posns)
    plt.xlabel("particle position")
    plt.title("Distribution of " + str(N) + " particle positions")

    plt.figure()
    plt.plot(posns[idxs_1][:,0],posns[idxs_2][:,0],'o')
    plt.xlabel("position of particle #1")
    plt.ylabel("position of particle #2")
    plt.title(str(no_samples)+ " sampled pairs of particles")

    plt.figure()
    plt.hist2d(posns[idxs_1][:,0], posns[idxs_2][:,0], bins=(20, 20), cmap=plt.cm.jet)
    plt.xlabel("position of particle #1")
    plt.ylabel("position of particle #2")
    plt.title(str(no_samples)+ " sampled pairs of particles")

    # Show
    plt.show()
