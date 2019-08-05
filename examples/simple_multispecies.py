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
    N = 100

    # Positions for two species
    posns = {}
    posns["A"] = (np.random.random(size=(N,dim))-0.5) * (2.0 * L)
    posns["B"] = (np.random.random(size=(N,dim))-0.5) * (2.0 * L)

    # Cutoff counting probabilities for particles that are more than:
    # std_dev_clip_mult * std_dev
    # away from each-other
    std_dev = 10.0
    std_dev_clip_mult = 3.0

    # Setup the sampler
    prob_calculator = ProbCalculatorMultiSpecies(posns_dict=posns, dim=dim, std_dev=std_dev, std_dev_clip_mult=std_dev_clip_mult)

    # Make the sampler
    sampler = SamplerMultiSpecies(prob_calculator)
    sampler.set_logging_level(logging.INFO)

    # Handle failure
    def handle_fail(code):
        print("Could not draw pair: code: %s" % code)
        sys.exit(0)

    # Sample pair using rejection sampling
    no_tries_max = 100
    code = sampler.rejection_sample(no_tries_max=no_tries_max)
    if code != ReturnCode.SUCCESS:
        handle_fail(code)

    # Sample pair using CDF
    code = sampler.cdf_sample()
    if code != ReturnCode.SUCCESS:
        handle_fail(code)
