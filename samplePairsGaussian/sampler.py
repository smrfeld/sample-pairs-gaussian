from .sampler_abstract_base import *
from .prob_calculator import *

import numpy as np

class Sampler(SamplerAbstractBase):
    """Sampler class.

    Attributes:
    idx_first_particle (int): idx of the first particle chosen
    idx_second_particle (int): idx of the second particle chosen
    """



    def __init__(self, prob_calculator):
        """Constructor.

        Args:
        prob_calculator (ProbCalculator): probability calculator
        """
        super().__init__(prob_calculator)

        # Init all other structures
        self.idx_first_particle = None
        self.idx_second_particle = None



    def rejection_sample(self, no_tries_max=100):
        """Use rejection sampling to sample the pair

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Reset
        self.idx_first_particle = None
        self.idx_second_particle = None

        success = super().rejection_sample(no_tries_max=no_tries_max)
        if success:
            self.idx_first_particle = self.prob_calculator.probs_idxs_first_particle[self.idx_chosen]
            self.idx_second_particle = self.prob_calculator.probs_idxs_second_particle[self.idx_chosen]
            self._logger.info("> samplePairsGaussian < Accepted pair idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle))
            return True
        else:
            return False



    def cdf_sample(self):
        """Sample by directly calculating the CDF via np.random.choice
        Ensures that the probabilities in the ProbCalculator are normalized before proceeding

        Returns:
        bool: True for success, False for failure
        """

        # Reset
        self.idx_first_particle = None
        self.idx_second_particle = None

        success = super().cdf_sample()
        if success:
            self.idx_first_particle = self.prob_calculator.probs_idxs_first_particle[self.idx_chosen]
            self.idx_second_particle = self.prob_calculator.probs_idxs_second_particle[self.idx_chosen]
            self._logger.info("> samplePairsGaussian < Accepted pair idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle))
            return True
        else:
            return False
