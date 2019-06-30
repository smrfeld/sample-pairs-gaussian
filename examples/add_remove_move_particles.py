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
    L = 5.0

    # Dimensionality (1D,2D,3D, etc. - all dims are supported, but only 1D will be possible to plot pairs and probability)
    dim = 1

    # Number of particles
    N = 3

    # Positions
    posns = (np.random.random(size=(N,dim))-0.5) * (2.0 * L)

    # Setup the sampler

    # Make the probability calculator
    std_dev = 3.0
    std_dev_clip_mult = None
    prob_calculator = ProbCalculator(posns=posns, dim=dim, std_dev=std_dev,std_dev_clip_mult=std_dev_clip_mult)

    # Print probs
    print("--- Initial probabilities ---")
    for i_pair in range(0,prob_calculator.no_idx_pairs_possible):
        idx1 = prob_calculator.probs_idxs_first_particle[i_pair]
        idx2 = prob_calculator.probs_idxs_second_particle[i_pair]
        print("particles: " + str(idx1) + " @ " + str(prob_calculator.posns[idx1]) + " and " + str(idx2) + " @ " + str(prob_calculator.posns[idx2]) + " dist: " + str(np.sqrt(prob_calculator.dists_squared[i_pair])) + " prob: " + str(prob_calculator.probs[i_pair]))

    # Add particle
    idx_new = 1
    posn_new = np.array([0.5])
    prob_calculator.add_particle(idx_new, posn_new)

    # Print probs
    print("--- After adding particle: idx: " + str(idx_new) + " posn: " + str(posn_new) + " ---")
    for i_pair in range(0,prob_calculator.no_idx_pairs_possible):
        idx1 = prob_calculator.probs_idxs_first_particle[i_pair]
        idx2 = prob_calculator.probs_idxs_second_particle[i_pair]
        print("particles: " + str(idx1) + " @ " + str(prob_calculator.posns[idx1]) + " and " + str(idx2) + " @ " + str(prob_calculator.posns[idx2]) + " dist: " + str(np.sqrt(prob_calculator.dists_squared[i_pair])) + " prob: " + str(prob_calculator.probs[i_pair]))

    # Remove particle
    idx_remove = 2
    prob_calculator.remove_particle(idx_remove)

    # Print probs
    print("--- After removing particle: idx: " + str(idx_remove) + " ---")
    for i_pair in range(0,prob_calculator.no_idx_pairs_possible):
        idx1 = prob_calculator.probs_idxs_first_particle[i_pair]
        idx2 = prob_calculator.probs_idxs_second_particle[i_pair]
        print("particles: " + str(idx1) + " @ " + str(prob_calculator.posns[idx1]) + " and " + str(idx2) + " @ " + str(prob_calculator.posns[idx2]) + " dist: " + str(np.sqrt(prob_calculator.dists_squared[i_pair])) + " prob: " + str(prob_calculator.probs[i_pair]))

    # Remove particle
    idx_move = 1
    posn_move = np.array([0.6])
    prob_calculator.move_particle(idx_move, posn_move)

    # Print probs
    print("--- After moving particle: idx: " + str(idx_move) + " from: " + str(prob_calculator.posns[idx_move]) +  " to new pos: " + str(posn_move) + " ---")
    for i_pair in range(0,prob_calculator.no_idx_pairs_possible):
        idx1 = prob_calculator.probs_idxs_first_particle[i_pair]
        idx2 = prob_calculator.probs_idxs_second_particle[i_pair]
        print("particles: " + str(idx1) + " @ " + str(prob_calculator.posns[idx1]) + " and " + str(idx2) + " @ " + str(prob_calculator.posns[idx2]) + " dist: " + str(np.sqrt(prob_calculator.dists_squared[i_pair])) + " prob: " + str(prob_calculator.probs[i_pair]))
