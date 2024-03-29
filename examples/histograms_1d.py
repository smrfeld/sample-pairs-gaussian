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
    N = 1000

    # Positions
    posns = (np.random.random(size=(N,dim))-0.5) * (2.0 * L)

    # Handle failure
    def handle_fail(code):
        print("Could not draw particle: code: %s" % code)
        sys.exit(0)

    # Setup the sampler

    # Make the probability calculator
    std_dev_tmp = 1.0
    prob_calculator = ProbCalculator(posns,dim,std_dev_tmp)
    # Distances have already been computed for us between all particles

    # Make the sampler
    sampler = Sampler(prob_calculator)

    # Func to get the samples
    def get_samples(no_samples, std_dev, std_dev_clip_mult):

        # Set std dev
        prob_calculator.set_std_dev(std_dev,std_dev_clip_mult)

        no_tries_max = 100
        idxs_1 = []
        idxs_2 = []
        for i in range(0,no_samples):

            # Sample using rejection sampling
            code = sampler.rejection_sample(no_tries_max=no_tries_max)
            if code != ReturnCode.SUCCESS:
                handle_fail(code)

            # Alternatively sample using CDF
            #success = sampler.cdf_sample_pair_given_nonzero_probs_for_first_particle()
            #if not success:
            #    handle_fail()

            # Append
            idxs_1.append(sampler.idx_first_particle)
            idxs_2.append(sampler.idx_second_particle)

        return [idxs_1, idxs_2]

    # Histogram of particle positions
    plt.figure()
    plt.hist(posns)
    plt.xlabel("particle position")
    plt.title("Distribution of " + str(N) + " particle positions")

    # No samples
    no_samples = 10000

    # Cutoff counting probabilities for particles that are more than:
    # std_dev_clip_mult * std_dev
    # away from each-other
    std_devs = [1.0,10.0,30.0]
    std_dev_clip_mult = 3.0
    for std_dev in std_devs:

        # Sample
        idxs_1, idxs_2 = get_samples(no_samples,std_dev,std_dev_clip_mult)

        # Plot
        plt.figure()
        idxs_1_symmetric = np.concatenate((idxs_1,idxs_2))
        idxs_2_symmetric = np.concatenate((idxs_2,idxs_1))
        plt.hist2d(posns[idxs_1_symmetric][:,0], posns[idxs_2_symmetric][:,0], bins=(20, 20), cmap=plt.cm.jet)
        plt.xlabel("position of particle #1")
        plt.ylabel("position of particle #2")
        plt.title("Std. dev. = " + str(std_dev))

    # Show
    plt.show()
