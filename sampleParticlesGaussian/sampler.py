from prob_calculator import *

import numpy as np

class Sampler:

    def __init__(self, posns, std_dev, std_dev_clip_mult):

        # Prob prob_calculator
        self.prob_calculator = ProbCalculator(posns,std_dev,std_dev_clip_mult)

    def rejection_sample_first_particle(self, no_tries_max=100, compute_probs=True):

        # Form probabilities
        if compute_probs:
            self.prob_calculator.compute_probs_first_particle()

        i_try = 0
        self.idx_first_particle = None
        while self.idx_first_particle == None and i_try < no_tries_max:
            i_try += 1

            # Random particle (uniform)
            idx = np.random.randint(self.prob_calculator.n)

            # Rejection sampling
            r = np.random.uniform(0.0,self.prob_calculator.max_prob_first_particle)
            if r < self.prob_calculator.probs_first_particle[idx]:
                # Accept
                self.idx_first_particle = idx
                print("Accepted first particle idx: " + str(idx) + " after: " + str(i_try) + " tries")

        if i_try == no_tries_max:
            print("Error! Could not sample the first particle after: " + str(no_tries_max) + "tries.")

    def rejection_sample_second_particle(self, no_tries_max=100, compute_probs=True):

        # Form probabilities
        if compute_probs:
            self.prob_calculator.compute_probs_second_particle(self.idx_first_particle)

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
                print("Accepted second particle idx: " + str(idx) + " after: " + str(i_try) + " tries")

        if i_try == no_tries_max:
            print("Error! Could not sample the second particle after: " + str(no_tries_max) + "tries.")

    def cdf_sample_first_particle(self,compute_probs=True):

        # Form probabilities
        if compute_probs:
            self.prob_calculator.compute_probs_first_particle(with_normalization=True)

        # Ensure normalized
        if self.prob_calculator.are_probs_first_particle_normalized == False:
            self.prob_calculator.are_probs_first_particle_normalized == True
            norm = np.sum(self.prob_calculator.probs_first_particle)
            self.prob_calculator.probs_first_particle /= norm
            self.max_prob_first_particle = 1.0

        # Choose
        self.idx_first_particle = np.random.choice(range(0,self.prob_calculator.n), 1, p=self.prob_calculator.probs_first_particle)[0]

        print("CDF sampled first particle idx: " + str(self.idx_first_particle))

    def cdf_sample_second_particle(self,compute_probs=True):

        # Form probabilities
        if compute_probs:
            self.prob_calculator.compute_probs_second_particle(self.idx_first_particle,with_normalization=True)

        # Ensure normalized
        if self.prob_calculator.are_probs_second_particle_normalized == False:
            self.prob_calculator.are_probs_second_particle_normalized == True
            norm = np.sum(self.prob_calculator.probs_second_particle)
            self.prob_calculator.probs_second_particle /= norm
            self.max_prob_second_particle = 1.0

        # Choose
        self.idx_second_particle = np.random.choice(self.prob_calculator.idxs_possible_second_particle, 1, p=self.prob_calculator.probs_second_particle)[0]

        print("CDF sampled second particle idx: " + str(self.idx_second_particle))
