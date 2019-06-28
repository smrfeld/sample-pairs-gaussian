from .prob_calculator import *

import numpy as np

class Sampler:
    """Sampler class.

    Attributes:
    prob_calculator (ProbCalculator): probability calculator
    idx_first_particle (int): idx of the first particle chosen
    idx_second_particle (int): idx of the second particle chosen

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
        self.idx_first_particle = None
        self.idx_second_particle = None



    def set_logging_level(self, level):
        """Sets the logging level

        Args:
        level (logging): logging level
        """
        self._logger.setLevel(level)



    def rejection_sample(self, std_dev, std_dev_clip_mult=3.0, no_tries_max=100):
        """Use rejection sampling to sample the pair

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Compute probs
        self.prob_calculator.compute_un_probs(std_dev, std_dev_clip_mult)

        # Check there are sufficient particles
        if self.prob_calculator.no_idx_pairs_possible == 0:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        return self.rejection_sample_given_nonzero_probs(no_tries_max=no_tries_max)



    def rejection_sample_given_nonzero_probs(self, no_tries_max=100):
        """Use rejection sampling, provided the probabilities are already calculated

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Reset
        self.idx_first_particle = None
        self.idx_second_particle = None

        i_try = 0
        while i_try < no_tries_max:
            i_try += 1

            # Random pair (uniform)
            idx_pair = np.random.randint(self.prob_calculator.no_idx_pairs_possible)

            # Rejection sampling
            r = np.random.uniform(0.0,self.prob_calculator.max_prob)
            if r < self.prob_calculator.probs[idx_pair]:
                # Accept
                self.idx_first_particle = self.prob_calculator.idxs_possible_first_particle[idx_pair]
                self.idx_second_particle = self.prob_calculator.idxs_possible_second_particle[idx_pair]
                self._logger.info("> samplePairsGaussian <")
                self._logger.info("Accepted pair idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle) + " after: " + str(i_try) + " tries")
                return True

        # Getting here means failure
        self._logger.info("> samplePairsGaussian <")
        self._logger.info("Fail: Could not rejection sample the pair after: " + str(i_try) + " tries.")
        return False



    def cdf_sample(self, std_dev, std_dev_clip_mult=3.0):
        """Sample by directly calculating the CDF via np.random.choice
        Ensures that the probabilities in the ProbCalculator are normalized before proceeding

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None

        Returns:
        bool: True for success, False for failure
        """

        self.prob_calculator.compute_un_probs(std_dev, std_dev_clip_mult)

        # Check there are sufficient particles
        if self.prob_calculator.no_idx_pairs_possible == 0:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        # Ensure normalized
        if self.prob_calculator.are_probs_normalized == False:
            self.prob_calculator.ensure_probs_are_normalized()

        return self.cdf_sample_given_nonzero_probs()


    def cdf_sample_given_nonzero_probs(self):
        """Sample given the probs already exist and are normalized

        Returns:
        bool: True for success, False for failure
        """

        # Choose
        idx_pair = np.random.choice(range(self.prob_calculator.no_idx_pairs_possible), 1, p=self.prob_calculator.probs)[0]
        self.idx_first_particle = self.prob_calculator.idxs_possible_first_particle[idx_pair]
        self.idx_second_particle = self.prob_calculator.idxs_possible_second_particle[idx_pair]

        self._logger.info("> samplePairsGaussian <")
        self._logger.info("CDF sampled pair idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle))

        return True
