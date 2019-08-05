from .prob_calculator import *

class ProbCalculatorMultiSpecies:
    """Calculates distances and probabilities for a set of particles of multiple species by holding a set of ProbCalculator objects.

    Attributes:
    posns_dict ({str: np.array([float])}): dict of species to posns for each species.
    dim (int): dimensionality of each point

    std_dev (float): standard deviation
    std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None

    dists_species (np.array([int])): species for the distances squared
    dists_idxs_first_particle (np.array([int])): idx of the first particle. Only unique combinations together with idxs 2
    dists_idxs_second_particle (np.array([int])): idx of the second particle. Only unique combinations together with idxs 1
    dists_squared (np.array(float)): distances squared between all particles
    no_dists (int): # distances possible

    probs_species (np.array([str])): species possible.
    probs_idxs_first_particle (np.array([int])): idx of the first particle. Only unique combinations together with idxs_2.
    probs_idxs_second_particle (np.array([int])): idx of the second particle. Only unique combinations together with idxs_1.
    probs (np.array([float])): probs of each pair of particles.
    no_idx_pairs_possible (int): # idx pairs possible
    max_prob (float): the maximum probability value, useful for rejection sampling

    Private attributes:
    _logger (logger): logging
    """

    def __init__(self, posns_dict, dim, std_dev, std_dev_clip_mult=3.0):
        """Constructor.

        Args:
        posns_dict ({str: np.array([float])}): dict of species to posns for each species.
        dim (int): dimensionality of each point
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None
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
        self._posns_dict = posns_dict
        self._dim = dim
        self._std_dev = std_dev
        self._std_dev_clip_mult = std_dev_clip_mult
        self._n_dict = {}
        self._n = 0
        for species, posns in self._posns_dict.items():
            self._n_dict[species] = len(posns)
            self._n += self._n_dict[species]

        # Initialize all manner of other properties for possible later use
        self._reset()

        # Compute probs
        self._compute_probs()



    def set_logging_level(self, level):
        """Sets the logging level
        Args:
        level (logging): logging level
        """
        self._logger.setLevel(level)



    def _reset(self):
        """Reset structures
        """

        self._dists_species = np.array([]).astype(str)
        self._dists_idxs_first_particle = np.array([]).astype(int)
        self._dists_idxs_second_particle = np.array([]).astype(int)
        self._dists_squared = np.array([]).astype(float)
        self._no_dists = 0

        self._probs_species = np.array([]).astype(str)
        self._probs_idxs_first_particle = np.array([]).astype(int)
        self._probs_idxs_second_particle = np.array([]).astype(int)
        self._probs = np.array([]).astype(float)
        self._max_prob = None
        self._norm = None
        self._no_idx_pairs_possible = 0



    # Getters
    @property
    def posns_dict(self):
        return self._posns_dict

    @property
    def n_dict(self):
        return self._n_dict

    @property
    def n(self):
        return self._n

    @property
    def dim(self):
        return self._dim

    @property
    def std_dev(self):
        return self._std_dev

    @property
    def std_dev_clip_mult(self):
        return self._std_dev_clip_mult

    @property
    def dists_species(self):
        return self._dists_species

    @property
    def dists_idxs_first_particle(self):
        return self._dists_idxs_first_particle

    @property
    def dists_idxs_second_particle(self):
        return self._dists_idxs_second_particle

    @property
    def dists_squared(self):
        return self._dists_squared

    @property
    def no_dists(self):
        return self._no_dists

    @property
    def probs_species(self):
        return self._probs_species

    @property
    def probs_idxs_first_particle(self):
        return self._probs_idxs_first_particle

    @property
    def probs_idxs_second_particle(self):
        return self._probs_idxs_second_particle

    @property
    def probs(self):
        return self._probs

    @property
    def max_prob(self):
        return self._max_prob

    @property
    def norm(self):
        return self._norm

    @property
    def no_idx_pairs_possible(self):
        return self._no_idx_pairs_possible



    def set_std_dev(self, std_dev, std_dev_clip_mult=3.0):
        """Set the std dev and clip multiplier, and recalculate the probabilities.

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None
        """

        self._std_dev = std_dev
        self._std_dev_clip_mult = std_dev_clip_mult

        # Distances are still valid; recompute probs
        self._compute_probs_from_dists()



    def _compute_probs(self):
        """Compute un-normalized probabilities for drawing the first particle out of n possible particles.
        """

        # Reset
        self._reset()

        for species, posns in self._posns_dict.items():

            # Make a prob calculator
            pc = ProbCalculator(posns=posns,dim=self._dim,std_dev=self._std_dev,std_dev_clip_mult=self._std_dev_clip_mult)

            # This has distances; reap them
            self._dists_squared = np.append(self._dists_squared,pc.dists_squared)
            self._dists_idxs_first_particle = np.append(self._dists_idxs_first_particle,pc.dists_idxs_first_particle)
            self._dists_idxs_second_particle = np.append(self._dists_idxs_second_particle,pc.dists_idxs_second_particle)
            self._dists_species = np.append(self._dists_species, np.full(pc.no_dists, species))

        # No idx pairs and dists
        self._no_dists = len(self._dists_squared)

        # Convert to probs
        self._compute_probs_from_dists()


    def _compute_probs_from_dists(self):
        """Compute normalized probabilities given distances squared
        """

        two_var = 2.0 * pow(self._std_dev,2)

        # Clip distances at std_dev_clip_mult * sigma
        if self._std_dev_clip_mult != None:
            max_dist_squared = pow(self._std_dev_clip_mult*self._std_dev,2)

            # Eliminate beyond max dist
            idxs_within = self._dists_squared < max_dist_squared
            self._probs_species = self._dists_species[idxs_within]
            self._probs_idxs_first_particle = self._dists_idxs_first_particle[idxs_within]
            self._probs_idxs_second_particle = self._dists_idxs_second_particle[idxs_within]
            dists_squared_filtered = self._dists_squared[idxs_within]

            # Compute gaussians
            self._probs = np.exp(- dists_squared_filtered / two_var) / pow(np.sqrt(np.pi * two_var),self._dim)

        else:

            # Take all idxs
            self._probs_species = np.copy(self._dists_species)
            self._probs_idxs_first_particle = np.copy(self._dists_idxs_first_particle)
            self._probs_idxs_second_particle = np.copy(self._dists_idxs_second_particle)

            # Compute gaussians
            self._probs = np.exp(- self._dists_squared / two_var) / pow(np.sqrt(np.pi * two_var),self._dim)

        # No idx pairs
        self._no_idx_pairs_possible = len(self._probs)

        # normalize
        self._norm = np.sum(self._probs)
        self._probs /= self._norm

        # Max
        if self._no_idx_pairs_possible > 0:
            self._max_prob = max(self._probs)
        else:
            self._max_prob = None



    def add_particle(self, species, idx, posn):
        """Add a particle

        Args:
        species (str): species
        idx (int): position at which to insert the particle
        posn (np.array([float])): position in d dimensions
        """

        self._posns_dict[species] = np.insert(self._posns_dict[species],idx,posn,axis=0)
        self._n_dict[species] += 1
        self._n += 1

        if self._n_dict[species] == 1:
            return # Finished

        # Shift idxs such that they do not refer to idx
        probs_idxs_all = np.arange(self._no_idx_pairs_possible)
        probs_idxs_this_species = self._probs_species == species
        shift_1 = probs_idxs_all[np.logical_and(probs_idxs_this_species,self._probs_idxs_first_particle >= idx)]
        self._probs_idxs_first_particle[shift_1] += 1
        shift_2 = probs_idxs_all[np.logical_and(probs_idxs_this_species,self._probs_idxs_second_particle >= idx)]
        self._probs_idxs_second_particle[shift_2] += 1

        dists_idxs_all = np.arange(self._no_dists)
        dists_idxs_this_species = self._dists_species == species
        shift_1 = dists_idxs_all[np.logical_and(dists_idxs_this_species,self._dists_idxs_first_particle >= idx)]
        self._dists_idxs_first_particle[shift_1] += 1
        shift_2 = dists_idxs_all[np.logical_and(self._dists_idxs_second_particle >= idx,dists_idxs_this_species)]
        self._dists_idxs_second_particle[shift_2] += 1

        # Idxs of particle pairs to add
        idxs_add_1 = np.full(self._n_dict[species]-1,idx)
        idxs_add_2 = np.delete(np.arange(self._n_dict[species]),idx)
        species_add = np.full(self._n_dict[species]-1,species)

        # Distances squared
        dr = self._posns_dict[species][idxs_add_1] - self._posns_dict[species][idxs_add_2]
        dists_squared_add = np.sum(dr*dr, axis=1)

        # Append to the dists
        self._dists_species = np.append(self._dists_species,species_add)
        self._dists_idxs_first_particle = np.append(self._dists_idxs_first_particle,idxs_add_1)
        self._dists_idxs_second_particle = np.append(self._dists_idxs_second_particle,idxs_add_2)
        self._dists_squared = np.append(self._dists_squared,dists_squared_add)
        self._no_dists += len(dists_squared_add)

        # Max dist
        if self._std_dev_clip_mult != None:
            max_dist_squared = pow(self._std_dev_clip_mult*self._std_dev,2)

            # Filter by max dist
            idxs_within = dists_squared_add < max_dist_squared
            idxs_add_1 = idxs_add_1[idxs_within]
            idxs_add_2 = idxs_add_2[idxs_within]
            species_add = species_add[idxs_within]
            dists_squared_add = dists_squared_add[idxs_within]

        # Compute gaussians
        two_var = 2.0 * pow(self._std_dev,2)
        probs_add = np.exp(- dists_squared_add / two_var) / pow(np.sqrt(np.pi * two_var),self._dim)

        # Normalization: scale existing probs back
        self._probs *= self._norm

        # Append
        self._probs_species = np.append(self._probs_species,species_add)
        self._probs_idxs_first_particle = np.append(self._probs_idxs_first_particle,idxs_add_1)
        self._probs_idxs_second_particle = np.append(self._probs_idxs_second_particle,idxs_add_2)
        self._probs = np.append(self._probs,probs_add)

        # Re-normalize
        self._norm += np.sum(probs_add)
        self._probs /= self._norm

        # Number of pairs now
        self._no_idx_pairs_possible += len(idxs_add_1)

        # Max prob
        if self._no_idx_pairs_possible > 0:
            self._max_prob = max(self._probs)
        else:
            self._max_prob = None



    def remove_particle(self, species, idx):
        """Remove a particle

        Args:
        species (str): species
        idx (int): idx of the particle to remove
        """

        self._posns_dict[species] = np.delete(self._posns_dict[species],idx,axis=0)
        self._n_dict[species] -= 1
        self._n -= 1

        if self._n_dict[species] == 0:
            return # Finished

        # Idxs to delete in the pair list
        probs_idxs_all = np.arange(self._no_idx_pairs_possible)
        probs_idxs_this_species = self._probs_species == species
        probs_idxs_delete_1 = probs_idxs_all[np.logical_and(probs_idxs_this_species, self._probs_idxs_first_particle == idx)]
        probs_idxs_delete_2 = probs_idxs_all[np.logical_and(probs_idxs_this_species, self._probs_idxs_second_particle == idx)]
        probs_idxs_delete = np.append(probs_idxs_delete_1,probs_idxs_delete_2)

        dists_idxs_all = np.arange(self._no_dists)
        dists_idxs_this_species = self._dists_species == species
        dists_idxs_delete_1 = dists_idxs_all[np.logical_and(dists_idxs_this_species, self._dists_idxs_first_particle == idx)]
        dists_idxs_delete_2 = dists_idxs_all[np.logical_and(dists_idxs_this_species, self._dists_idxs_second_particle == idx)]
        dists_idxs_delete = np.append(dists_idxs_delete_1,dists_idxs_delete_2)

        # Normalization: scale existing probs back
        self._probs *= self._norm

        # New normalization after removing pairs
        self._norm -= np.sum(self._probs[probs_idxs_delete])

        # Remove all probs associated with this
        self._probs = np.delete(self._probs,probs_idxs_delete)
        self._probs_idxs_first_particle = np.delete(self._probs_idxs_first_particle,probs_idxs_delete)
        self._probs_idxs_second_particle = np.delete(self._probs_idxs_second_particle,probs_idxs_delete)
        self._probs_species = np.delete(self._probs_species,probs_idxs_delete)

        self._dists_squared = np.delete(self._dists_squared,dists_idxs_delete)
        self._dists_idxs_first_particle = np.delete(self._dists_idxs_first_particle,dists_idxs_delete)
        self._dists_idxs_second_particle = np.delete(self._dists_idxs_second_particle,dists_idxs_delete)
        self._dists_species = np.delete(self._dists_species,dists_idxs_delete)
        self._no_dists -= len(dists_idxs_delete)

        # Re-normalize
        self._probs /= self._norm

        # Number of pairs now
        self._no_idx_pairs_possible -= len(probs_idxs_delete)

        # Max prob
        if self._no_idx_pairs_possible > 0:
            self._max_prob = max(self._probs)
        else:
            self._max_prob = None

        # Shift the idxs such that they again include idx
        probs_idxs_all = np.arange(self._no_idx_pairs_possible)
        probs_idxs_this_species = self._probs_species == species
        shift_1 = probs_idxs_all[np.logical_and(probs_idxs_this_species, self._probs_idxs_first_particle > idx)]
        self._probs_idxs_first_particle[shift_1] -= 1
        shift_2 = probs_idxs_all[np.logical_and(probs_idxs_this_species, self._probs_idxs_second_particle > idx)]
        self._probs_idxs_second_particle[shift_2] -= 1

        dists_idxs_all = np.arange(self._no_dists)
        dists_idxs_this_species = self._dists_species == species
        shift_1 = dists_idxs_all[np.logical_and(dists_idxs_this_species,self._dists_idxs_first_particle > idx)]
        self._dists_idxs_first_particle[shift_1] -= 1
        shift_2 = dists_idxs_all[np.logical_and(self._dists_idxs_second_particle > idx,dists_idxs_this_species)]
        self._dists_idxs_second_particle[shift_2] -= 1



    def move_particle(self, species, idx, new_posn):
        """Move a particle

        Args:
        idx (int): idx of the particle to move
        new_posn (np.array([float])): new position in d dimensions
        """

        # Remove and reinsert
        self.remove_particle(species, idx)
        self.add_particle(species, idx, new_posn)



    def compute_gaussian_sum_between_particle_and_existing(self, species, posn, excluding_idxs=[]):
        """Compute normalization = sum_{j} exp( -(xi-xj)^2 / 2*sigma^2 ) for a given particle xi and all other existing particles, possibly excluding some idxs

        Args:
        species (str): the species
        posn (np.array([float])): position of the particle
        excluding_idxs ([int]): list of particle idxs in [0,n) to exclude

        Returns:
        float: the sum, else None
        """

        if self._n_dict[species] == 0:
            return 0.0

        # Exclude idxs
        idxs = np.array(range(0,self._n_dict[species]))
        if excluding_idxs != []:
            idxs = np.delete(idxs,excluding_idxs)
        posns = self._posns_dict[species][idxs]

        if len(posns) == 0:
            return 0.0

        # Distances squared
        dr = posns - posn
        dists_squared = np.sum(dr*dr, axis=1)

        # Max dist
        if self._std_dev_clip_mult != None:
            max_dist_squared = pow(self._std_dev_clip_mult*self._std_dev,2)

            # Filter by max dist
            stacked = np.array([idxs,dists_squared]).T
            idxs, dists_squared = stacked[stacked[:,1] < max_dist_squared].T

        # Compute gaussians
        two_var = 2.0 * pow(self._std_dev,2)
        gauss = np.exp(- dists_squared / two_var) / pow(np.sqrt(np.pi * two_var),self._dim)

        # Normalization
        return np.sum(gauss)
