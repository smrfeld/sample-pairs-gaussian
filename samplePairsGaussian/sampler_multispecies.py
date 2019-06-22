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



    def sample_species(self, std_dev, std_dev_clip_mult):
        """Sample the species

        Args:
        compute_probs_species (bool): whether to first call compute_probs_species for the ProbCalculatorMultiSpecies
        compute_probs (bool): whether to first call compute_probs_first_particle for the ProbCalculatorMultiSpecies
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff

        Returns:
        bool: True for success, False for failure
        """

        # Form probabilities
        self.prob_calculator_multispecies.compute_un_probs_first_particle_for_all_species(std_dev, std_dev_clip_mult)

        # Form probabilities for species
        self.prob_calculator_multispecies.compute_species_probs()

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
        if self.prob_calculator_multispecies.no_species > 0:
            self.idx_species_particles = np.random.choice(range(0,self.prob_calculator_multispecies.no_species), 1, p=self.prob_calculator_multispecies.probs_species)[0]
            self.species_particles = self.prob_calculator_multispecies.species_arr[self.idx_species_particles]
            return True
        else:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: Could not sample the species because no species has enough particles.")
            return False



    def rejection_sample_first_particle(self, std_dev, std_dev_clip_mult, no_tries_max=100):
        """Use rejection sampling to sample the first particle

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Form probabilities
        self.prob_calculator_multispecies.compute_un_probs_first_particle_for_all_species(std_dev, std_dev_clip_mult)

        # Form probabilities for species
        self.prob_calculator_multispecies.compute_species_probs()

        # Check there are sufficient particles
        if self.prob_calculator_multispecies.no_species_possible == 0:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        return self.rejection_sample_first_particle_given_nonzero_probs(no_tries_max=no_tries_max)



    def rejection_sample_first_particle_given_nonzero_probs(self, no_tries_max=100):
        """Use rejection sampling to sample the first particle, given nonzero probs for the first particle and species

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        # Reset current
        self.idx_first_particle = None

        i_try = 0
        while i_try < no_tries_max:
            i_try += 1

            # Random species
            success = self.sample_species_given_nonzero_probs()
            if not success:
                return False # No species available

            # Rejection sample
            sampler = self.samplers[self.idx_species_particles]
            success = sampler.rejection_sample_first_particle_given_nonzero_probs(no_tries_max=1)
            if success:
                self.idx_first_particle = sampler.idx_first_particle

                self._logger.info("> samplePairsGaussian <")
                self._logger.info("Accepted first particle species: " + str(self.species_particles) + " idx: " + str(self.idx_first_particle) + " after: " + str(i_try) + " tries")

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

        # Reset current
        self.idx_second_particle = None

        sampler = self.samplers[self.idx_species_particles]
        success = sampler.rejection_sample_second_particle(no_tries_max=no_tries_max)
        if success:
            self.idx_second_particle = sampler.idx_second_particle

            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Accepted second particle species: " + str(self.species_particles) + " idx: " + str(self.idx_second_particle))

            return True

        else:

            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: Could not sample the second particle.")

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

        # Form probabilities
        self.prob_calculator_multispecies.compute_un_probs_first_particle_for_all_species(std_dev, std_dev_clip_mult)

        # Form probabilities for species
        self.prob_calculator_multispecies.compute_species_probs()

        # Check there are sufficient particles
        if self.prob_calculator_multispecies.no_species_possible == 0:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        return self.rejection_sample_pair_given_nonzero_probs_for_first_particle(no_tries_max=no_tries_max)



    def rejection_sample_pair_given_nonzero_probs_for_first_particle(self, no_tries_max=100):
        """Use rejection sampling to sample a pair of particles

        Args:
        no_tries_max (int): max. no. of tries for rejection sampling

        Returns:
        bool: True for success, False for failure
        """

        i_try = 0
        while i_try < no_tries_max:
            i_try += 1

            success = self.rejection_sample_first_particle_given_nonzero_probs(no_tries_max=1)
            if not success:
                continue # try again

            success = self.rejection_sample_second_particle(no_tries_max=1)
            if not success:
                continue # try again

            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Accepted pair particles species: " + str(self.species_particles) + " idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle) + " after: " + str(i_try) + " tries")

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

        # Form probabilities
        self.prob_calculator_multispecies.compute_un_probs_first_particle_for_all_species(std_dev, std_dev_clip_mult)

        # Form probabilities for species
        self.prob_calculator_multispecies.compute_species_probs()

        # Check there are sufficient particles
        if self.prob_calculator_multispecies.no_species_possible == 0:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        # Ensure normalized
        self.prob_calculator_multispecies.ensure_probs_first_particle_are_normalized_for_all_species()

        return self.cdf_sample_first_particle_given_nonzero_probs()



    def cdf_sample_first_particle_given_nonzero_probs(self):
        """Sample the first particle by directly calculating the CDF via np.random.choice
        Ensures that the probabilities in the ProbCalculator are normalized before proceeding

        Returns:
        bool: True for success, False for failure
        """

        # Choose species
        self.sample_species_given_nonzero_probs()

        # Choose particle
        sampler = self.samplers[self.idx_species_particles]
        success = sampler.cdf_sample_first_particle_given_nonzero_probs()
        if success:
            self.idx_first_particle = sampler.idx_first_particle

            self._logger.info("> samplePairsGaussian <")
            self._logger.info("CDF sampled first particle species: " + str(self.species_particles) + " idx: " + str(self.idx_first_particle))

            return True
        else:
            return False



    def cdf_sample_second_particle(self):
        """Sample the second particle by directly calculating the CDF via np.random.choice
        Ensures that the probabilities in the ProbCalculator are normalized before proceeding

        Returns:
        bool: True for success, False for failure
        """

        sampler = self.samplers[self.idx_species_particles]
        success = sampler.cdf_sample_second_particle()
        if success:
            self.idx_second_particle = sampler.idx_second_particle

            self._logger.info("> samplePairsGaussian <")
            self._logger.info("CDF sampled second particle species: " + str(self.species_particles) + " idx: " + str(self.idx_second_particle))

            return True
        else:
            return False



    def cdf_sample_pair(self, std_dev, std_dev_clip_mult):
        """Sample both particles directly using numpy.random.choice

        Args:
        std_dev (float): standard deviation
        std_dev_clip_mult (float): multiplier for the standard deviation cutoff

        Returns:
        bool: True for success, False for failure
        """

        # Form probabilities
        self.prob_calculator_multispecies.compute_un_probs_first_particle_for_all_species(std_dev, std_dev_clip_mult)

        # Form probabilities for species
        self.prob_calculator_multispecies.compute_species_probs()

        # Check there are sufficient particles
        if self.prob_calculator_multispecies.no_species_possible == 0:
            self._logger.info("> samplePairsGaussian <")
            self._logger.info("Fail: not enough particles within the cutoff radius to sample a pair.")
            return False

        # Ensure normalized
        self.prob_calculator_multispecies.ensure_probs_first_particle_are_normalized_for_all_species()

        return self.cdf_sampler_pair_given_nonzero_probs_for_first_particle()



    def cdf_sampler_pair_given_nonzero_probs_for_first_particle(self):
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

        # Set logging back
        self._logger.info("> samplePairsGaussian <")
        self._logger.info("CDF sampled pair particle species: " + str(self.species_particles) + " idxs: " + str(self.idx_first_particle) + " " + str(self.idx_second_particle))

        # Done
        return True
