from .return_codes import *
from abc import ABC, abstractmethod

import numpy as np
import logging

class SamplerAbstractBase(ABC):
    """Sampler class.

    Attributes:
    prob_calculator (ProbCalculator): probability calculator
    idx_chosen (int): the idx chosen

    Private attributes:
    _logger (logger): logging
    """



    def __init__(self, prob_calculator):
        """Constructor. Also computes distances.

        Args:
        prob_calculator (ProbCalculator): probability calculator
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

        # Prob prob_calculator
        self.prob_calculator = prob_calculator

        # Init all other structures
        self.idx_chosen = None



    def set_logging_level(self, level):
        """Sets the logging level

        Args:
        level (logging): logging level
        """
        self._logger.setLevel(level)



    @abstractmethod
    def rejection_sample(self, no_tries_max=100):
        """Use rejection sampling to sample the pair

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        ReturnCode: return code
        """

        # Check there are sufficient pairs
        if self.prob_calculator.n == 0:
            return ReturnCode.FAIL_ZERO_PARTICLES
        elif self.prob_calculator.n == 1:
            return ReturnCode.FAIL_ONE_PARTICLE
        elif self.prob_calculator.no_idx_pairs_possible == 0:
            return ReturnCode.FAIL_STD_CLIP_MULT

        # Reset
        self.idx_chosen = None

        i_try = 0
        while i_try < no_tries_max:
            i_try += 1

            # Random pair (uniform)
            idx_pair = np.random.randint(self.prob_calculator.no_idx_pairs_possible)

            # Rejection sampling
            r = np.random.uniform(0.0,self.prob_calculator.max_prob)
            if r < self.prob_calculator.probs[idx_pair]:
                # Accept
                self.idx_chosen = idx_pair
                self._logger.info("> samplePairsGaussian < [base] Accepted idx: " + str(self.idx_chosen) + " after: " + str(i_try) + " tries")
                return ReturnCode.SUCCESS

        # Getting here means failure
        self._logger.info("> samplePairsGaussian < [base] Fail: Could not rejection sample the pair after: " + str(i_try) + " tries.")
        return ReturnCode.FAIL_NO_SAMPLING_TRIES



    @abstractmethod
    def cdf_sample(self):
        """Sample by directly calculating the CDF via np.random.choice

        Returns:
        ReturnCode: return code
        """

        # Check there are sufficient pairs
        if self.prob_calculator.n == 0:
            return ReturnCode.FAIL_ZERO_PARTICLES
        elif self.prob_calculator.n == 1:
            return ReturnCode.FAIL_ONE_PARTICLE
        elif self.prob_calculator.no_idx_pairs_possible == 0:
            return ReturnCode.FAIL_STD_CLIP_MULT

        # Choose
        self.idx_chosen = np.random.choice(range(self.prob_calculator.no_idx_pairs_possible), 1, p=self.prob_calculator.probs)[0]

        self._logger.info("> samplePairsGaussian < [base] CDF sampled idx: " + str(self.idx_chosen))

        return ReturnCode.SUCCESS
