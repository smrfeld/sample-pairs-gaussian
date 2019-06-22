from .prob_calculator import *

class ProbCalculatorMultiSpecies:
    """Calculates distances and probabilities for a set of particles of multiple species by holding a set of ProbCalculator objects.

    Attributes:
    prob_calculator_arr ([ProbCalculator]): list of prob calculators.
    species_arr ([str]): list of species names/labels
    no_species (int): the number of species

    Private attributes:
    _logger (logger): logging
    """

    def __init__(self, prob_calculator_arr, species_arr):
        """Constructor.

        Args:
        prob_calculator_arr ([ProbCalculator]): list of prob calculators.
        species_arr ([str]): list of species names/labels
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

        # Length
        if len(species_arr) != len(prob_calculator_arr):
            self._logger.info("> samplePairsGaussian <")
            self._logger.error("Error: length of species array = " + str(len(species_arr)) + " does not match the probabilities: " + str(len(prob_calculator_arr)) + "; Quitting.")
            sys.exit(1)

        # vars
        self.no_species = len(species_arr)
        self.no_species_possible = self.no_species
        self.prob_calculator_arr = prob_calculator_arr
        self.species_arr = species_arr

        # compute species probabilities
        self.compute_species_probs()



    def set_logging_level(self, level):
        """Sets the logging level
        Args:
        level (logging): logging level
        """
        self._logger.setLevel(level)



    def compute_un_probs_first_particle_for_all_species(self, std_dev, std_dev_clip_mult):
        """Compute un-normalized probabilities for drawing the first particle out of n possible particles.
        """
        for prob_calculator in self.prob_calculator_arr:
            prob_calculator.compute_un_probs_first_particle(std_dev, std_dev_clip_mult)



    def ensure_probs_first_particle_are_normalized_for_all_species(self):
        """Normalize the probs for the first particle
        """
        for prob_calculator in self.prob_calculator_arr:
            if not prob_calculator.are_probs_first_particle_normalized:
                prob_calculator.normalize_probs_first_particle()



    def get_prob_calculator_for_species(self, species):
        """Get prob calculator for a species

        Args:
        species (str): the species
        """
        idx_species = self.species_arr.index(species)
        return self.prob_calculator_arr[idx_species]



    def compute_species_probs(self):
        """Compute probabilities for the different species.
        """

        self.no_species_possible = self.no_species
        self.probs_species = np.zeros(self.no_species)
        for i in range(0,self.no_species):
            # Reject if there are not at least 2 particles
            no_idxs_possible_first_particle = self.prob_calculator_arr[i].no_idxs_possible_first_particle
            if no_idxs_possible_first_particle >= 2:
                self.probs_species[i] = no_idxs_possible_first_particle
            else:
                self.probs_species[i] = 0
        n_total = np.sum(self.probs_species)

        if n_total > 0.:
            self.probs_species /= n_total
        else:
            self.probs_species = np.zeros(self.no_species)
            self.no_species_possible = 0
