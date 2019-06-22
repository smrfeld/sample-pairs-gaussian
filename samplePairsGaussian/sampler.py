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



    def rejection_sample_first_particle(self, std_dev, std_dev_clip_mult, no_tries_max=100):
        """Use rejection sampling to sample the first particle

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Compute probs
        self.prob_calculator.compute_un_probs_first_particle(std_dev, std_dev_clip_mult)

        # Check there are sufficient particles
        if self.prob_calculator.no_idxs_possible_first_particle < 2:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        return self.rejection_sample_first_particle_given_nonzero_probs(no_tries_max=no_tries_max)



    def rejection_sample_first_particle_given_nonzero_probs(self, no_tries_max=100):
        """Use rejection sampling to sample the first particle, provided the probabilities are already calculated

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        i_try = 0
        self.idx_first_particle = None
        while self.idx_first_particle == None and i_try < no_tries_max:
            i_try += 1

            # Random particle (uniform)
            idx = np.random.randint(self.prob_calculator.no_idxs_possible_first_particle)

            # Rejection sampling
            r = np.random.uniform(0.0,self.prob_calculator.max_prob_first_particle)
            if r < self.prob_calculator.probs_first_particle[idx]:
                # Accept
                self.idx_first_particle = self.prob_calculator.idxs_possible_first_particle[idx]
                self._logger.info("> samplePairsGaussian <")
                self._logger.info("Accepted first particle idx: " + str(self.idx_first_particle) + " after: " + str(i_try) + " tries")
                return True

        # Getting here means failure
        self._logger.info("> samplePairsGaussian <")
        self._logger.info("Fail: Could not sample the first particle after: " + str(i_try) + " tries.")
        return False



    def rejection_sample_second_particle(self, no_tries_max=100):
        """Use rejection sampling to sample the second particle

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Compute probs
        self.prob_calculator.compute_un_probs_second_particle(self.idx_first_particle)

        i_try = 0
        self.idx_second_particle = None
        while self.idx_second_particle == None and i_try < no_tries_max:
            i_try += 1

            # Random particle (uniform)
            idx = np.random.randint(self.prob_calculator.no_idxs_possible_second_particle)

            # Rejection sampling
            r = np.random.uniform(0.0,self.prob_calculator.max_prob_second_particle)
            if r < self.prob_calculator.probs_second_particle[idx]:
                # Accept
                self.idx_second_particle = self.prob_calculator.idxs_possible_second_particle[idx]

                self._logger.info("> samplePairsGaussian <")
                self._logger.info("Accepted second particle idx: " + str(self.idx_second_particle) + " after: " + str(i_try) + " tries")

                return True

        # Getting here means failure
        self._logger.info("> samplePairsGaussian <")
        self._logger.info("Fail: Could not sample the second particle after: " + str(no_tries_max) + " tries.")
        return False



    def rejection_sample_pair(self, std_dev, std_dev_clip_mult, no_tries_max=100):
        """Use rejection sampling to sample a pair of particles

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Calculate probs
        self.prob_calculator.compute_un_probs_first_particle(std_dev, std_dev_clip_mult)

        # Check there are sufficient particles
        if self.prob_calculator.no_idxs_possible_first_particle < 2:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        return self.rejection_sample_pair_given_nonzero_probs_for_first_particle(no_tries_max=no_tries_max)



    def rejection_sample_pair_given_nonzero_probs_for_first_particle(self, no_tries_max=100):
        """Use rejection sampling to sample a pair of particles, given nonzero probabilities for the first particle

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Turn off logging temp
        level = self._logger.level
        self._logger.setLevel(logging.CRITICAL)

        i_try = 0
        while i_try < no_tries_max:
            i_try += 1

            success = self.rejection_sample_first_particle_given_nonzero_probs(no_tries_max=1)
            if not success:
                continue # try again

            success = self.rejection_sample_second_particle(no_tries_max=1)
            if not success:
                continue # try again

            # Set logging back
            self._logger.setLevel(level)

            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Accepted pair particles idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle) + " after: " + str(i_try) + " tries")

            # Done
            return True

        # Getting here means failure
        self._logger.info("> samplePairsGaussian <")
        self._logger.info("Fail: Could not sample the two particles after: " + str(no_tries_max) + " tries.")
        return False



    def cdf_sample_first_particle(self, std_dev, std_dev_clip_mult):
        """Sample the first particle by directly calculating the CDF via np.random.choice
        Ensures that the probabilities in the ProbCalculator are normalized before proceeding

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff

        Returns:
        bool: True for success, False for failure
        """

        self.prob_calculator.compute_un_probs_first_particle(std_dev, std_dev_clip_mult)

        # Check there are sufficient particles
        if self.prob_calculator.no_idxs_possible_first_particle < 2:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        # Ensure normalized
        if self.prob_calculator.are_probs_first_particle_normalized == False:
            self.prob_calculator.normalize_probs_first_particle()

        return self.cdf_sample_first_particle_given_nonzero_probs()


    def cdf_sample_first_particle_given_nonzero_probs(self):
        """Sample the first particle, given the probs already exist and are normalized

        Returns:
        bool: True for success, False for failure
        """

        # Choose
        self.idx_first_particle = np.random.choice(self.prob_calculator.idxs_possible_first_particle, 1, p=self.prob_calculator.probs_first_particle)[0]

        self._logger.info("> samplePairsGaussian <")
        self._logger.info("CDF sampled first particle idx: " + str(self.idx_first_particle))

        return True



    def cdf_sample_second_particle(self):
        """Sample the second particle by directly calculating the CDF via np.random.choice
        Ensures that the probabilities in the ProbCalculator are normalized before proceeding

        Returns:
        bool: True for success, False for failure
        """

        # Form probabilities
        self.prob_calculator.compute_un_probs_second_particle(self.idx_first_particle)

        # Check there are sufficient particles
        if self.prob_calculator.no_idxs_possible_second_particle < 2:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        # Ensure normalized
        if self.prob_calculator.are_probs_second_particle_normalized == False:
            self.prob_calculator.normalize_probs_second_particle()

        # Choose
        self.idx_second_particle = np.random.choice(self.prob_calculator.idxs_possible_second_particle, 1, p=self.prob_calculator.probs_second_particle)[0]

        self._logger.info("> samplePairsGaussian <")
        self._logger.info("CDF sampled second particle idx: " + str(self.idx_second_particle))

        return True



    def cdf_sample_pair(self, std_dev, std_dev_clip_mult):
        """Sample both particles directly using numpy.random.choice

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff

        Returns:
        bool: True for success, False for failure
        """

        # Compute probs
        self.prob_calculator.compute_un_probs_first_particle(std_dev, std_dev_clip_mult)

        # Check there are sufficient particles
        if self.prob_calculator.no_idxs_possible_first_particle < 2:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        # Ensure normalized
        if self.prob_calculator.are_probs_first_particle_normalized == False:
            self.prob_calculator.normalize_probs_first_particle()

        return self.cdf_sample_pair_given_nonzero_probs_for_first_particle()



    def cdf_sample_pair_given_nonzero_probs_for_first_particle(self):
        """Sample both particles directly using numpy.random.choice

        Returns:
        bool: True for success, False for failure
        """

        success = self.cdf_sample_first_particle_given_nonzero_probs()
        if not success:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: Could not sample the two particles using cdf_sample_pair.")
            return False

        success = self.cdf_sample_second_particle()
        if not success:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: Could not sample the two particles using cdf_sample_pair.")
            return False

        self._logger.info("> samplePairsGaussian <")
        self._logger.info("CDF sampled pair particle idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle))

        # Done
        return True
