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

    # Positions of two species
    posns = {}
    posns["A"] = (np.random.random(size=(N,dim))-0.5) * (2.0 * L)
    posns["B"] = (np.random.random(size=(N,dim))-0.5) * (2.0 * L)

    # Setup the sampler

    # Make the probability calculator
    std_dev = 10.0
    std_dev_clip_mult = 3.0
    prob_calculator = ProbCalculatorMultiSpecies(posns,dim,std_dev,std_dev_clip_mult)

    # Make the sampler
    sampler = SamplerMultiSpecies(prob_calculator)
    sampler.set_logging_level(logging.INFO)

    # Sample many particle pairs

    # Handle failure
    def handle_fail(code):
        print("Could not draw particle: code: %s" % code)
        sys.exit(0)

    no_samples = 1000
    no_tries_max = 100
    idxs_1 = {"A": [], "B": []}
    idxs_2 = {"A": [], "B": []}
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
        idxs_1[sampler.species_particles].append(sampler.idx_first_particle)
        idxs_2[sampler.species_particles].append(sampler.idx_second_particle)

    idxs_1["A"] = np.array(idxs_1["A"]).astype(int)
    idxs_1["B"] = np.array(idxs_1["B"]).astype(int)
    idxs_2["A"] = np.array(idxs_2["A"]).astype(int)
    idxs_2["B"] = np.array(idxs_2["B"]).astype(int)

    # Plot

    plt.figure()
    plt.hist(posns["A"])
    plt.xlabel("particle position")
    plt.title("Distribution of " + str(N) + " particle positions of species A")

    plt.figure()
    plt.hist(posns["B"])
    plt.xlabel("particle position")
    plt.title("Distribution of " + str(N) + " particle positions of species B")

    plt.figure()
    # Make symmetric
    idxs_1_symmetric = np.concatenate((idxs_1["A"],idxs_2["A"]))
    idxs_2_symmetric = np.concatenate((idxs_2["A"],idxs_1["A"]))
    plt.plot(posns["A"][idxs_1_symmetric][:,0],posns["A"][idxs_2_symmetric][:,0],'ro')
    idxs_1_symmetric = np.concatenate((idxs_1["B"],idxs_2["B"]))
    idxs_2_symmetric = np.concatenate((idxs_2["B"],idxs_1["B"]))
    plt.plot(posns["B"][idxs_1_symmetric][:,0],posns["B"][idxs_2_symmetric][:,0],'bo')
    plt.xlabel("position of particle #1")
    plt.ylabel("position of particle #2")
    plt.title(str(no_samples)+ " sampled pairs of particles")

    # Show
    plt.show()
