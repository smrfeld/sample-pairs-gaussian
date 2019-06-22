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

    # Dimensionality (1D,2D,3D, etc. - all dims are supported, but only 1D will be possible to plot pairs and probability)
    dim = 2

    # Number of particles
    N = 1000

    # Positions
    posns = np.random.normal(0.0,10.0, size=(N,dim))

    # Setup the sampler

    # Make the probability calculator
    prob_calculator = ProbCalculator(posns)

    # Make the sampler
    sampler = Sampler(prob_calculator)

    # Sample many particle pairs

    # Handle failure
    def handle_fail(ith_particle):
        print("Could not draw the " + str(ith_particle) + " particle: try adjusting the std. dev. for the probability cutoff.")
        sys.exit(0)

    # Cutoff counting probabilities for particles that are more than:
    # std_dev_clip_mult * std_dev
    # away from each-other
    std_dev = 1.0
    std_dev_clip_mult = 3.0

    # For efficiency, just compute the first particle probability now
    prob_calculator.compute_un_probs_first_particle(std_dev,std_dev_clip_mult)

    no_samples = 10000
    no_tries_max = 100
    idxs_1 = []
    idxs_2 = []
    for i in range(0,no_samples):

        # Sample using rejection sampling
        success = sampler.rejection_sample_pair_given_nonzero_probs_for_first_particle(no_tries_max=no_tries_max)
        if not success:
            handle_fail(i)

        # Alternatively sample using CDF
        #success = sampler.cdf_sample_pair_given_nonzero_probs_for_first_particle()
        #if not success:
        #    handle_fail()

        # Append
        idxs_1.append(sampler.idx_first_particle)
        idxs_2.append(sampler.idx_second_particle)

    # For each idx, get distances
    dists = {}
    for i in range(0,no_samples):
        idx1 = idxs_1[i]
        idx2 = idxs_2[i]
        pos1 = posns[idx1]
        pos2 = posns[idx2]
        dist = np.sqrt(sum(pow(pos1-pos2,2)))
        if idx1 in dists:
            dists[idx1].append(dist)
        else:
            dists[idx1] = [dist]
        if idx2 in dists:
            dists[idx2].append(dist)
        else:
            dists[idx2] = [dist]

    # Average dist
    ave_dist = np.full(N,0.0)
    for idx, dist in dists.items():
        ave_dist[idx] = np.mean(dist)

    # Counts
    counts = np.full(N,0).astype(int)
    for idx, dist in dists.items():
        counts[idx] = len(dist)

    # Plot
    plt.figure()
    min_dist = min(ave_dist)
    max_dist = max(ave_dist-min_dist)
    cols = [(dist-min_dist) / max_dist for dist in ave_dist]
    plt.scatter(posns[:,0],posns[:,1], c=cols, cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(str(N)+ " particles: ave dist of other particle")

    plt.figure()
    min_count = min(counts)
    max_count = max(counts-min_count)
    cols = [(count-min_count) / max_count for count in counts]
    plt.scatter(posns[:,0],posns[:,1], c=cols, cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(str(N)+ " particles: counts of draws")

    # Show
    plt.show()
