from .prob_calculator_multispecies import *
from .sampler import *

import numpy as np

class SamplerMultiSpecies:
    """Sampler for multiple species.

    Attributes:
    prob_calculator_multispecies (ProbCalculatorMultiSpecies): probability calculator
    samplers ([Sampler]): sampler for each species
    idx_first_particle (int): idx of the first particle chosen
    idx_second_particle (int): idx of the second particle chosen
    idx_species_particles (int): idx of the species of both particles
    species_particles (str): species of both particles

    Private attributes:
    _logger (logger): logging
    """



    def __init__(self, prob_calculator_multispecies):
        """Constructor. Also computes distances.

        Args:
        prob_calculator_multispecies (ProbCalculatorMultiSpecies): probability calculator for multiple species
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
        self.prob_calculator_multispecies = prob_calculator_multispecies

        self.samplers = []
        for prob_calculator in self.prob_calculator_multispecies.prob_calculator_arr:
            self.samplers.append(Sampler(prob_calculator))

        # Init all other structures
        self.idx_species_particles = None
        self.species_particles = None
        self.idx_first_particle = None
        self.idx_second_particle = None



    def set_logging_level(self, level):
        """Sets the logging level

        Args:
        level (logging): logging level
        """
        self._logger.setLevel(level)



    def sample_species(self, std_dev, std_dev_clip_mult=3.0):
        """Sample the species

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None

        Returns:
        bool: True for success, False for failure
        """

        # Form probabilities
        self.prob_calculator_multispecies.compute_un_probs_for_all_species(std_dev, std_dev_clip_mult)

        # Sample
        return self.sample_species_given_nonzero_probs()


    def sample_species_given_nonzero_probs(self):
        """Sample the species

        Returns:
        bool: True for success, False for failure
        """

        # Reset current
        self.idx_species_particles = None
        self.species_particles = None

        # Sample
        self.idx_species_particles = np.random.choice(range(0,self.prob_calculator_multispecies.no_species), 1, p=self.prob_calculator_multispecies.probs_species)[0]
        self.species_particles = self.prob_calculator_multispecies.species_arr[self.idx_species_particles]

        return True



    def rejection_sample(self, std_dev, std_dev_clip_mult=3.0, no_tries_max=100):
        """Use rejection sampling

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff, else None
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Form probabilities
        self.prob_calculator_multispecies.compute_un_probs_for_all_species(std_dev, std_dev_clip_mult)

        return self.rejection_sample_given_nonzero_probs(no_tries_max=no_tries_max)



    def rejection_sample_given_nonzero_probs(self, no_tries_max=100):
        """Use rejection sampling, given nonzero probs

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Reset current
        self.idx_first_particle = None
        self.idx_second_particle = None

        i_try = 0
        while i_try < no_tries_max:
            i_try += 1

            # Random species
            success = self.sample_species_given_nonzero_probs()
            if not success:
                return False # No species available

            # Rejection sample
            sampler = self.samplers[self.idx_species_particles]
            success = sampler.rejection_sample_given_nonzero_probs(no_tries_max=1)
            if success:
                self.idx_first_particle = sampler.idx_first_particle
                self.idx_second_particle = sampler.idx_second_particle

                self._logger.info("> samplePairsGaussian <")
                self._logger.info("Accepted particle pair of species: " + str(self.species_particles) + " idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle) + " after: " + str(i_try) + " tries")

                return True

        # Getting here means failure
        self._logger.info("> samplePairsGaussian <")
        self._logger.info("Fail: Could not rejection sample after: " + str(i_try) + " tries.")
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

        # Form probabilities
        self.prob_calculator_multispecies.compute_un_probs_for_all_species(std_dev, std_dev_clip_mult)

        # Ensure normalized
        self.prob_calculator_multispecies.ensure_probs_are_normalized_for_all_species()

        return self.cdf_sample_given_nonzero_probs()



    def cdf_sample_given_nonzero_probs(self):
        """Sample by directly calculating the CDF via np.random.choice

        Returns:
        bool: True for success, False for failure
        """

        # Random species
        success = self.sample_species_given_nonzero_probs()
        if not success:
            return False # No species available

        # Choose particle
        sampler = self.samplers[self.idx_species_particles]
        success = sampler.cdf_sample_given_nonzero_probs()
        if success:
            self.idx_first_particle = sampler.idx_first_particle
            self.idx_second_particle = sampler.idx_second_particle

            self._logger.info("> samplePairsGaussian <")
            self._logger.info("CDF sampled particles of species: " + str(self.species_particles) + " idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle))

            return True
        else:

            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Failed to sample pair with CDF")
            return False
