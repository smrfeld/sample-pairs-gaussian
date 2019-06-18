import numpy as np

class ProbCalculator:

    def __init__(self, posns, std_dev, std_dev_clip_mult):

        # vars
        self.posns = posns
        self.n = len(self.posns)
        self.std_dev = std_dev
        self.std_dev_clip_mult = std_dev_clip_mult

        # Compute distances
        self.compute_dists_squared()

    def compute_dists_squared(self):

        # uti is a list of two (1-D) numpy arrays
        # containing the indices of the upper triangular matrix
        self._uti = np.triu_indices(self.n,k=1)        # k=1 eliminates diagonal indices

        # uti[0] is i, and uti[1] is j from the previous example
        dr = self.posns[self._uti[0]] - self.posns[self._uti[1]]            # computes differences between particle positions
        self.dists_squared = np.sum(dr*dr, axis=1)    # computes distances squared; D is a 4950 x 1 np array

    def compute_probs_first_particle(self, with_normalization=False):

        # Clip distances at std_dev_clip_mult * sigma
        max_dist_squared = pow(self.std_dev_clip_mult*self.std_dev,2)
        two_var = 2.0 * pow(self.std_dev,2)

        # Eliminate beyond max dist
        stacked = np.array([self._uti[0],self._uti[1],self.dists_squared]).T
        self._uti0filter, self._uti1filter, dists_squared_filter = stacked[stacked[:,2] < max_dist_squared].T

        # Compute gaussians
        self._exps = np.exp(-dists_squared_filter/two_var)

        # Not normalized probs for first particle
        self.probs_first_particle = np.bincount(self._uti0filter.astype(int),self._exps,minlength=self.n) + np.bincount(self._uti1filter.astype(int),self._exps,minlength=self.n)

        # Normalization
        if with_normalization:
            self.are_probs_first_particle_normalized = True
            norm = np.sum(self.probs_first_particle)
            self.probs_first_particle /= norm
            self.max_prob_first_particle = 1.0
        else:
            self.are_probs_first_particle_normalized = False
            self.max_prob_first_particle = max(self.probs_first_particle)

    def compute_probs_second_particle(self, idx_first_particle, with_normalization=False):

        # Not normalized probs for second particle probs
        true_false_0 = self._uti0filter == idx_first_particle
        true_false_1 = self._uti1filter == idx_first_particle

        # Idxs
        self.idxs_possible_second_particle = np.concatenate((self._uti1filter[true_false_0], self._uti0filter[true_false_1])).astype(int)
        self.no_idxs_possible_second_particle = len(self.idxs_possible_second_particle)

        # Probs
        self.probs_second_particle = np.concatenate((self._exps[true_false_0],self._exps[true_false_1]))

        # Normalization
        if with_normalization:
            self.are_probs_second_particle_normalized = True
            norm = np.sum(self.probs_second_particle)
            self.probs_second_particle /= norm
        else:
            self.are_probs_second_particle_normalized = False
            self.max_prob_second_particle = max(self.probs_second_particle)
