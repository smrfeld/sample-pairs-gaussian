from .return_codes import *
from .sampler_abstract_base import *

import numpy as np

class SamplerMultiSpecies(SamplerAbstractBase):
    """Sampler for multiple species.

    Attributes:
    idx_first_particle (int): idx of the first particle chosen
    idx_second_particle (int): idx of the second particle chosen
    species_particles (str): species of both particles
    """



    def __init__(self, prob_calculator_multispecies):
        """Constructor.

        Args:
        prob_calculator_multispecies (ProbCalculatorMultiSpecies): probability calculator for multiple species
        """
        super().__init__(prob_calculator_multispecies)

        # Init all other structures
        self.species_particles = None
        self.idx_first_particle = None
        self.idx_second_particle = None



    def rejection_sample(self, no_tries_max=100):
        """Use rejection sampling to sample the pair

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        ReturnCode: return code
        """

        # Reset
        self.species_particles = None
        self.idx_first_particle = None
        self.idx_second_particle = None

        code = super().rejection_sample(no_tries_max=no_tries_max)
        if code == ReturnCode.SUCCESS:
            self.species_particles = self.prob_calculator.probs_species[self.idx_chosen]
            self.idx_first_particle = self.prob_calculator.probs_idxs_first_particle[self.idx_chosen]
            self.idx_second_particle = self.prob_calculator.probs_idxs_second_particle[self.idx_chosen]
            self._logger.info("> samplePairsGaussian < Accepted pair species: " + str(self.species_particles) + " idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle))

        return code



    def cdf_sample(self):
        """Sample by directly calculating the CDF via np.random.choice
        Ensures that the probabilities in the ProbCalculator are normalized before proceeding

        Returns:
        ReturnCode: return code
        """

        # Reset
        self.species_particles = None
        self.idx_first_particle = None
        self.idx_second_particle = None

        code = super().cdf_sample()
        if code == ReturnCode.SUCCESS:
            self.species_particles = self.prob_calculator.probs_species[self.idx_chosen]
            self.idx_first_particle = self.prob_calculator.probs_idxs_first_particle[self.idx_chosen]
            self.idx_second_particle = self.prob_calculator.probs_idxs_second_particle[self.idx_chosen]
            self._logger.info("> samplePairsGaussian < Accepted pair species: " + str(self.species_particles) + " idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle))

        return code
