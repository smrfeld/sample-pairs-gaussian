# If installed:
from samplePairsGaussian import *

# Else: Add the path to the module
# import sys
# sys.path.append('../samplePairsGaussian/')
# from sampler import *

import sys
import numpy as np

if __name__ == "__main__":

    # Generate random positions of particles

    # Uniformly spread particles from -L to L
    L = 100

    # Dimensionality (1D,2D,3D, etc. - all dims are supported, but only 1D will be possible to plot pairs and probability)
    dim = 1

    # Number of particles
    N = 0

    # Positions
    posns = (np.random.random(size=(N,dim))-0.5) * (2.0 * L)

    # Setup the sampler

    # Cutoff counting probabilities for particles that are more than:
    # std_dev_clip_mult * std_dev
    # away from each-other
    std_dev = 10.0
    std_dev_clip_mult = None

    # Make the probability calculator
    prob_calculator = ProbCalculator(posns,dim,std_dev,std_dev_clip_mult)

    # Make the sampler
    sampler = Sampler(prob_calculator)
    sampler.set_logging_level(logging.INFO)

    # Handle failure
    def handle_fail(code):
        print("Could not draw pair: code: %s" % code)

    # Sample pair using rejection sampling
    no_tries_max = 100
    code = sampler.rejection_sample(no_tries_max=no_tries_max)
    if code != ReturnCode.SUCCESS:
        handle_fail(code)

    # Sample pair using CDF
    code = sampler.cdf_sample()
    if code != ReturnCode.SUCCESS:
        handle_fail(code)

    # Sum
    gaussian_sum = prob_calculator.compute_gaussian_sum_between_particle_and_existing(np.random.random(size=3), excluding_idxs=[])
    print("Gaussian sum between random particle and others: %f" % gaussian_sum)
