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
            self._logger.error("Error: length of species array = " + str(len(species_arr)) + " does not match the probabilities: " + str(len(prob_calculator_arr)) + "; Quitting.")
            sys.exit(0)

        # vars
        self.no_species = len(species_arr)
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



    def compute_species_probs(self):
        """Compute probabilities for the different species.
        """

        self.probs_species = np.zeros(self.no_species)
        for i in range(0,self.no_species):
            self.probs_species[i] = self.prob_calculator_arr[i].n
        self.n_total = np.sum(self.probs_species)
        self.probs_species /= self.n_total
