import sys
import copy
import numpy as np
import random
from Chromosome import Chromosome

ordering_first_stage_options  = ['LPT', 'SPT', 'LNS', 'SNS']#, 'Random','Random','Random','Random']
ordering_second_stage_options = ['LPT', 'SRT', 'LNS', 'SNS', 'AVA1','AVA2','Random','Random','Random','Random','Random','Random']


class Population:
    def __init__(self, m1, m2, jobs_1, jobs_2, pop_size=100):
        self._best_fitness = 0
        self._best_individual_index = 0
        self._mean_fitness = 0
        self._std_fitness = 0

        self._best_fitness_2 = 0
        self._best_individual_index_2 = 0
        self._mean_fitness_2 = 0
        self._std_fitness_2 = 0
        self._best_fit_iter = 0
        self._pop_size = pop_size
        self._generation = 0
        self._m1 = m1
        self._m2 = m2
        self._jobs_1 = jobs_1
        self._jobs_2 = jobs_2
        self._population = None
        self.create_initial_population()

    def print_population(self):
        for i in (self.population):
            print(i)

    def print_stats(self):
        print("BF(1): "+str(self._best_fitness)+" | MF(1): "+ str(self._mean_fitness) + " | SF(1): " + str(self._std_fitness))
        print("BF(2): " + str(self._best_fitness_2) + " | MF(2): " + str(self._mean_fitness_2) + " | SF(2): " + str(self._std_fitness_2) + "Iter:" + str(self._best_fit_iter))

    @property
    def best_individual_2(self):
        return self._population[self._best_individual_index_2]

    @property
    def population(self):
        return self._population

    @property
    def best_fitness(self):
        return self._best_fitness

    @property
    def best_individual_index(self):
        return self._best_individual_index

    @property
    def best_individual_index_2(self):
        return self._best_individual_index_2

    @property
    def std_fitness(self):
        return self._std_fitness

    @property
    def std_fitness_2(self):
        return self._std_fitness_2

    @property
    def mean_fitness(self):
        return self._mean_fitness

    @property
    def mean_fitness_2(self):
        return self._mean_fitness_2

    @property
    def best_fitness_2(self):
        return self._best_fitness_2

    @property
    def generation(self):
        return self._generation

    @population.setter
    def population(self, value):
        self._population = value

    def compute_stats(self, iter):
        min_makespan_1 = sys.maxsize
        min_makespan_2 = sys.maxsize

        min_index = 0
        min_index_2 = 0

        makespans = []
        makespans_2 = []

        for i in range(len(self._population)):

            #self.population.population[i].compute_makespan_1()
            # population[i].print_first_stage()

            # Get the best so far...
            if min_makespan_1 > self._population[i]._makespan_1:
                min_makespan_1 = self._population[i]._makespan_1
                min_index = i

            if min_makespan_2 > self._population[i]._makespan_2:
                min_makespan_2 = self._population[i]._makespan_2
                min_index_2 = i


            makespans.append(self._population[i]._makespan_1)
            makespans_2.append(self._population[i]._makespan_2)

        #stats
        if self._best_fitness != min_makespan_1:
            self._best_fitness = min_makespan_1
            self._best_individual_index = min_index
            self._mean_fitness = np.mean(makespans)
            self._std_fitness = np.std(makespans)

        # stats
        if self._best_fitness_2 != min_makespan_2:
            self._best_fitness_2 = min_makespan_2
            self._best_individual_index_2 = min_index_2
            self._mean_fitness_2 = np.mean(makespans_2)
            self._std_fitness_2 = np.std(makespans_2)
            self._best_fit_iter = iter

    def update_stats(self):
        for p in self._population:
            p.update_stats()

    def compute_makespan(self):
        for i in range(len(self._population)):
            self._population[i].compute_makespan_1()

        self.compute_release_date()

        for i in range(len(self._population)):
            self._population[i].compute_makespan_2()

    def create_initial_population(self):
        """ Returns a population of chromossomes
            where each chromossome are the combination
            of two orderings (first + second stage) """
        self._population = [Chromosome(p, self._m1, self._m2, copy.deepcopy(self._jobs_1), copy.deepcopy(self._jobs_2))
                      for p in range(self._pop_size)]

        self._population[0].apply_lpt_rule_1()
        self._population[1].apply_spt_rule_1()
        self._population[2].apply_lns_rule_1()
        self._population[3].apply_sns_rule_1()

        # Shuffle job list of the first and second stage
        for i in range(0, len(self._population)):
            '''
            if random.random() < 0.3:
                ordering = random.choice(ordering_first_stage_options)
                if ordering == 'LPT':  # LPT
                    self._population[i].apply_lpt_rule_1()
                elif ordering == 'SPT':  # SRT
                    self._population[i].apply_spt_rule_1()
                elif ordering == 'LNS':  # LNS
                    self._population[i].apply_lns_rule_1()
                elif ordering == 'SNS':  # SNS
                    self._population[i].apply_sns_rule_1()
                #else:  # Random
                #    self._population[i].apply_random_rule_1()
            else:
            '''
            if i > 3:
                self._population[i].apply_random_rule_1()

            self._population[i].compute_makespan_1() # Computes makespan of first stage
            self._population[i].compute_release_date() # Computes release date of each second stage job
            self._population[i].apply_available_1_rule_2()  # Sort the second stage jobs in the shortest release date first

        for i in range(len(self._population)):
            self._population[i].compute_makespan_2() # Computes makespan of second stage

        return self._population
    '''
    def compute_release_date(self):
        for i in range(len(self._population)):
            for job2 in self._population[i].jobs_list_2:
                f_stage = self._population[i].jobs_list_1
                job2.release_date = max([j.release_date for j in f_stage if j.id in job2.predecessors])
    '''