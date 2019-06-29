import numpy as np
import logging
import sys

class ProbCalculator:
    """Calculates distances and probabilities for a set of particles.

    Attributes:
    posns (np.array([[float]])): particle positions. First axis are particles, seconds are coordinates in n-dimensional space
    n (int): number of particles

    idxs_possible_first_particle (np.array([int])): idx of the first particle. Only unique combinations together with idxs_2.
    idxs_possible_second_particle (np.array([int])): idx of the second particle. Only unique combinations together with idxs_1.
    probs (np.array([float])): probs of each pair of particles.
    no_idx_pairs_possible (int): # idx pairs possible
    are_probs_normalized (bool): whether the probabilities are normalized.
    max_prob (float): the maximum probability value, useful for rejection sampling

    Private attributes:
    _logger (logger): logging
    """



    def __init__(self, posns):
        """Constructor.

        Args:
        posns (np.array([[float]])): particle positions. First axis are particles, seconds are coordinates in n-dimensional space
        """

        # Setup the logger
        self._logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

        # Level of logging to display
        self._logger.setLevel(logging.ERROR)

        # vars
        self.posns = posns
        self.n = len(self.posns)

        # Initialize all manner of other properties for possible later use
        self.idxs_possible_first_particle = np.array([]).astype(int)
        self.idxs_possible_second_particle = np.array([]).astype(int)
        self.probs = np.array([]).astype(float)
        self.are_probs_normalized = False
        self.max_prob = None
        self.norm = None


    def set_logging_level(self, level):
        """Sets the logging level

        Args:
        level (logging): logging level
        """
        self._logger.setLevel(level)



    def add_particle(self, idx, posn, std_dev, std_dev_clip_mult):
        """Add a particle

        Args:
        idx (int): position at which to insert the particle
        posn (np.array([float])): position in d dimensions
        std_dev (float): standard deviation for re-computing the distances
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None
        """

        self.posns = np.insert(self.posns,idx,posn,axis=0)
        self.n += 1

        # Shift idxs such that they do not refer to idx
        idxs_all = np.arange(self.no_idx_pairs_possible)
        shift_1 = idxs_all[self.idxs_possible_first_particle >= idx]
        self.idxs_possible_first_particle[shift_1] += 1
        shift_2 = idxs_all[self.idxs_possible_second_particle >= idx]
        self.idxs_possible_second_particle[shift_2] += 1

        # Idxs of particle pairs to add
        idxs_add_1 = np.full(self.n-1,idx)
        idxs_add_2 = np.delete(np.arange(self.n),idx)

        # Distances squared
        dr = self.posns[idxs_add_1] - self.posns[idxs_add_2]
        dists_squared = np.sum(dr*dr, axis=1)

        # Max dist
        if std_dev_clip_mult != None:
            max_dist_squared = pow(std_dev_clip_mult*std_dev,2)

            # Filter by max dist
            stacked = np.array([idxs_add_1,idxs_add_2,dists_squared]).T
            idxs_add_1,idxs_add_2, dists_squared = stacked[stacked[:,2] < max_dist_squared].T

        # Compute gaussians
        two_var = 2.0 * pow(std_dev,2)
        dim = len(posn)
        probs_add = np.exp(- dists_squared / two_var) / pow(np.sqrt(np.pi * two_var),dim)

        # Append
        self.idxs_possible_first_particle = np.append(self.idxs_possible_first_particle,idxs_add_1)
        self.idxs_possible_second_particle = np.append(self.idxs_possible_second_particle,idxs_add_2)
        self.probs = np.append(self.probs,probs_add)

        # Number of pairs now
        self.no_idx_pairs_possible = len(self.idxs_possible_first_particle)

        # If normalized, keep normalized!
        if self.are_probs_normalized:
            self.probs *= self.norm # scale back
            self.norm = np.sum(self.probs) # new norm
            self.probs /= self.norm # normalize

        # Max prob
        if self.no_idx_pairs_possible > 0:
            self.max_prob = max(self.probs)
        else:
            self.max_prob = None



    def remove_particle(self, idx):
        """Remove a particle

        Args:
        idx (int): idx of the particle to remove
        """

        self.posns = np.delete(self.posns,idx,axis=0)
        self.n -= 1

        # Remove all probs associated with this
        idxs_all = np.arange(self.no_idx_pairs_possible)
        idxs_delete_1 = idxs_all[self.idxs_possible_first_particle == idx]
        idxs_delete_2 = idxs_all[self.idxs_possible_second_particle == idx]
        idxs_delete = np.concatenate((idxs_delete_1,idxs_delete_2))
        self.probs = np.delete(self.probs,idxs_delete)
        self.idxs_possible_first_particle = np.delete(self.idxs_possible_first_particle,idxs_delete)
        self.idxs_possible_second_particle = np.delete(self.idxs_possible_second_particle,idxs_delete)

        # Number of pairs now
        self.no_idx_pairs_possible = len(self.idxs_possible_first_particle)

        # If normalized, keep normalized!
        if self.are_probs_normalized:
            self.probs *= self.norm # scale back
            self.norm = np.sum(self.probs) # new norm
            self.probs /= self.norm # normalize

        # Max prob
        if self.no_idx_pairs_possible > 0:
            self.max_prob = max(self.probs)
        else:
            self.max_prob = None

        # Shift the idxs such that they again include idx
        idxs_all = np.arange(self.no_idx_pairs_possible)
        shift_1 = idxs_all[self.idxs_possible_first_particle > idx]
        self.idxs_possible_first_particle[shift_1] -= 1
        shift_2 = idxs_all[self.idxs_possible_second_particle > idx]
        self.idxs_possible_second_particle[shift_2] -= 1



    def move_particle(self, idx, new_posn, std_dev, std_dev_clip_mult):
        """Move a particle

        Args:
        idx (int): idx of the particle to move
        new_posn (np.array([float])): new position in d dimensions
        std_dev (float): standard deviation for re-computing the distances
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None
        """

        # Remove and reinsert
        self.remove_particle(idx)
        self.add_particle(idx, new_posn, std_dev, std_dev_clip_mult)



    def compute_un_probs(self, std_dev, std_dev_clip_mult):
        """Compute un-normalized probabilities.

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None
        """

        # Check there are sufficient particles
        if self.n < 2:
            self.idxs_possible_first_particle = np.array([]).astype(int)
            self.idxs_possible_second_particle = np.array([]).astype(int)
            self.probs = np.array([]).astype(float)
            self.are_probs_normalized = False
            self.max_prob = None
            return

        # uti is a list of two (1-D) numpy arrays
        # containing the indices of the upper triangular matrix
        self.idxs_possible_first_particle, self.idxs_possible_second_particle = np.triu_indices(self.n,k=1)        # k=1 eliminates diagonal indices
        self.no_idx_pairs_possible = len(self.idxs_possible_first_particle)

        # uti[0] is i, and uti[1] is j from the previous example
        dr = self.posns[self.idxs_possible_first_particle] - self.posns[self.idxs_possible_second_particle]            # computes differences between particle positions
        dists_squared = np.sum(dr*dr, axis=1)    # computes distances squared; D is a 4950 x 1 np array

        # Clip distances at std_dev_clip_mult * sigma
        if std_dev_clip_mult != None:
            max_dist_squared = pow(std_dev_clip_mult*std_dev,2)

            # Eliminate beyond max dist
            stacked = np.array([self.idxs_possible_first_particle,self.idxs_possible_second_particle,dists_squared]).T
            self.idxs_possible_first_particle, self.idxs_possible_second_particle, dists_squared = stacked[stacked[:,2] < max_dist_squared].T
            self.idxs_possible_first_particle = self.idxs_possible_first_particle.astype(int)
            self.idxs_possible_second_particle = self.idxs_possible_second_particle.astype(int)
            self.no_idx_pairs_possible = len(self.idxs_possible_first_particle)

        # Compute gaussians
        two_var = 2.0 * pow(std_dev,2)
        dim = len(self.posns[0])
        self.probs = np.exp(- dists_squared / two_var) / pow(np.sqrt(np.pi * two_var),dim)

        # Normalized
        self.norm = None
        self.are_probs_normalized = False

        # Max
        if self.no_idx_pairs_possible > 0:
            self.max_prob = max(self.probs)
        else:
            self.max_prob = None



    def compute_gaussian_sum_between_particle_and_existing(self, posn, std_dev, std_dev_clip_mult=3.0, excluding_idxs=[]):
        """Compute normalization = sum_{j} exp( -(xi-xj)^2 / 2*sigma^2 ) for a given particle xi and all other existing particles, possibly excluding some idxs

        Args:
        posn (np.array([float])): position of the particle
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None
        excluding_idxs ([int]): list of particle idxs in [0,n) to exclude
        """

        if self.n == 0:
            return None

        # Exclude idxs
        idxs = np.array(range(0,self.n))
        if excluding_idxs != []:
            idxs = np.delete(idxs,excluding_idxs)
        posns = self.posns[idxs]

        if len(posns) == 0:
            return None

        # Distances squared
        dr = posns - posn
        dists_squared = np.sum(dr*dr, axis=1)

        # Max dist
        if std_dev_clip_mult != None:
            max_dist_squared = pow(std_dev_clip_mult*std_dev,2)

            # Filter by max dist
            stacked = np.array([idxs,dists_squared]).T
            idxs, dists_squared = stacked[stacked[:,1] < max_dist_squared].T

        # Compute gaussians
        two_var = 2.0 * pow(std_dev,2)
        dim = len(self.posns[0])
        gauss = np.exp(- dists_squared / two_var) / pow(np.sqrt(np.pi * two_var),dim)

        # Normalization
        return np.sum(gauss)



    def ensure_probs_are_normalized(self):
        """Normalize the probs
        """

        if not self.are_probs_normalized and self.no_idx_pairs_possible > 0:
            self.are_probs_normalized = True
            self.norm = np.sum(self.probs)
            self.probs /= self.norm
            self.max_prob = max(self.probs)
