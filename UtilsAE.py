import copy
import numpy as np
import random
from Population import Population
from Chromosome import Chromosome


class UtilsAE:

    def insertion_mutation(chromosome):
        new_chromosome = copy.deepcopy(chromosome)

        k = random.choice(range(len(new_chromosome.jobs_list_1)))

        i = random.choice(range(len(new_chromosome.jobs_list_1)))
        while i == k:
            i = random.choice(range(len(new_chromosome.jobs_list_1)))
        new_chromosome.jobs_list_1[i], new_chromosome.jobs_list_1[k] = new_chromosome.jobs_list_1[k], new_chromosome.jobs_list_1[i]

        return new_chromosome

    def inversion_mutation(chromosome):
        def partial_reverse(list_, from_, to):
            for i in range(0, int((to - from_) / 2)):
                (list_[from_ + i], list_[to - i]) = (list_[to - i], list_[from_ + i])

        new_chromosome = copy.deepcopy(chromosome)

        k = random.choice(range(len(new_chromosome.jobs_list_1-1)))
        size = random.choice(range(1, len(new_chromosome.jobs_list_1)-k))

        partial_reverse(new_chromosome.jobs_list_1, k, k+size)

        return new_chromosome

    def mutation_roll(chromosome):
        new_chromosome = copy.deepcopy(chromosome)

        new_chromosome.jobs_list_1 = np.roll(new_chromosome.jobs_list_1, random.randint(1,len(new_chromosome.jobs_list_1)-1))
        new_chromosome.jobs_list_1 = new_chromosome.jobs_list_1.tolist()

        return new_chromosome

    def mutation_slice_front(chromosome):
        #assert size < len(chromosome.jobs_list_1), "size must be < length(chromosome)"
        size = random.randint(1, len(chromosome.jobs_list_1)//2)

        new_chromosome = copy.deepcopy(chromosome)

        k = random.choice(range(len(chromosome.jobs_list_1)-size))
        new_chromosome.jobs_list_1 = chromosome.jobs_list_1[k:k+size]
        new_chromosome.jobs_list_1.extend(chromosome.jobs_list_1[0:k])
        new_chromosome.jobs_list_1.extend(chromosome.jobs_list_1[k+size:])

        return new_chromosome


    def mutation_slice_back(chromosome):
        #assert size < len(chromosome.jobs_list_1), "size must be < length(chromosome)"
        size = random.randint(1, len(chromosome.jobs_list_1)//2)

        new_chromosome = copy.deepcopy(chromosome)

        k = random.choice(range(len(chromosome.jobs_list_1)-size))
        new_chromosome.jobs_list_1=(chromosome.jobs_list_1[0:k])
        new_chromosome.jobs_list_1.extend(chromosome.jobs_list_1[k+size:])
        new_chromosome.jobs_list_1.extend(chromosome.jobs_list_1[k:k + size])

        return new_chromosome

    def mutation_slice_random(chromosome):
        #assert size < len(chromosome.jobs_list_1), "size must be < length(chromosome)"
        size = random.randint(1, len(chromosome.jobs_list_1)//2)

        new_chromosome = copy.deepcopy(chromosome)

        k = random.choice(range(len(new_chromosome.jobs_list_1)-size))

        backup = new_chromosome.jobs_list_1[k:k + size]
        del new_chromosome.jobs_list_1[k:k + size]
        pos = random.randint(0, len(new_chromosome.jobs_list_1))
        new_chromosome.jobs_list_1[pos:pos] = backup

        return new_chromosome

    def mutation_partition(chromosome):
        new_chromosome = copy.deepcopy(chromosome)
        k = random.choice(range(len(new_chromosome.jobs_list_1)-1))
        new_chromosome.jobs_list_1 = new_chromosome.jobs_list_1[k:] + new_chromosome.jobs_list_1[:k]
        return new_chromosome

    def uniform_selection(population, total_chromosomes=3):
        return random.sample(population, total_chromosomes)

    def tournament_selection(population, total_chromosomes_tournament=3, total_chromosomes=1):
        assert total_chromosomes <= total_chromosomes_tournament <= len(population)
        chromosomes_selected = set()

        while len(chromosomes_selected) < total_chromosomes:
            tournament_pop = random.sample(range(len(population)), total_chromosomes)
            tournament_pop = list(set(tournament_pop) - chromosomes_selected)
            if tournament_pop is not []:
                min_index = np.argmin([population[i].makespan_2 for i in tournament_pop])
                chromosomes_selected.add(min_index)

        return [population[i] for i in chromosomes_selected]

    def tournament_selection_survive(population, total_chromosomes_tournament=3, total_chromosomes=1):
        assert total_chromosomes <= total_chromosomes_tournament <= len(population)
        chromosomes_selected = set()

        while len(chromosomes_selected) < total_chromosomes:
            tournament_pop = random.sample(range(len(population)), total_chromosomes)
            tournament_pop = list(set(tournament_pop) - chromosomes_selected)
            if tournament_pop is not []:
                max_index = np.argmax([population[i].makespan_2 for i in tournament_pop])
                chromosomes_selected.add(max_index)

        return list(chromosomes_selected)

    def roulette_wheel_selection(chromosomes, total_chromosomes=1):
        chromosomes_list = []
        for i in range(total_chromosomes):
            max = sum(1/chromosome.makespan_2 for chromosome in chromosomes)
            pick = random.uniform(0, max)
            current = 0
            for chromosome in chromosomes:
                current += chromosome.makespan_2
                if current > pick:
                    chromosomes_list.append(chromosome)

        return chromosomes_list

    def roulette_wheel_selection_survive(chromosomes, total_chromosomes=1):
        chromosomes_list = []
        for i in range(total_chromosomes):
            max = sum(chromosome.makespan_2 for chromosome in chromosomes)
            pick = random.uniform(0, max)
            current = 0
            for chromosome in chromosomes:
                current += chromosome.makespan_2
                if current > pick:
                    chromosomes_list.append(chromosomes.index(chromosome))
                    if len(chromosomes_list) == total_chromosomes: return chromosomes_list

    def apply_pmx_crossover(chromosome_1, chromosome_2):
        child_1, child_2 = copy.deepcopy(chromosome_1), copy.deepcopy(chromosome_2)

        size = random.randint(1, len(chromosome_2.jobs_list_1) // 2)
        k = random.randint(0, len(chromosome_2.jobs_list_1) - size)

        new_child_1 = [None for i in range(len(chromosome_1.jobs_list_1))]

        for i in range(k, k + size):
            new_child_1[i] = child_1.jobs_list_1[i]

        for i in range(k, k + size):
            if child_2.jobs_list_1[i] not in [j for j in new_child_1 if j is not None]:

                ind = child_2.jobs_list_1.index(child_1.jobs_list_1[i])
                while k <= ind < k + size:
                    ind = child_2.jobs_list_1.index(child_1.jobs_list_1[ind])

                new_child_1[ind] = child_2.jobs_list_1[i]

        for i in range(len(chromosome_1.jobs_list_1)):
            if new_child_1[i] is None:
                new_child_1[i] = child_2.jobs_list_1[i]

        new_child_2 = [None for i in range(len(chromosome_2.jobs_list_1))]

        for i in range(k, k + size):
            new_child_2[i] = child_2.jobs_list_1[i]

        for i in range(k, k + size):
            if child_1.jobs_list_1[i] not in [j for j in new_child_2 if j is not None]:

                ind = child_1.jobs_list_1.index(child_2.jobs_list_1[i])
                while k <= ind < k + size:
                    ind = child_1.jobs_list_1.index(child_2.jobs_list_1[ind])

                new_child_2[ind] = child_1.jobs_list_1[i]

        for i in range(len(chromosome_2.jobs_list_1)):
            if new_child_2[i] is None:
                new_child_2[i] = child_1.jobs_list_1[i]

        child_1.jobs_list_1 = new_child_1[:]
        child_2.jobs_list_1 = new_child_2[:]

        return child_1, child_2

    def apply_ox_crossover(chromosome_1, chromosome_2):
        child_1, child_2 = copy.deepcopy(chromosome_1), copy.deepcopy(chromosome_2)

        size = random.randint(1, len(chromosome_2.jobs_list_1) // 2)
        k = random.randint(0, len(chromosome_2.jobs_list_1) - size)

        backup = child_1.jobs_list_1[k:k + size]

        j = k + size
        for i in range(k + size, k + size + len(chromosome_1.jobs_list_1)):
            if child_2.jobs_list_1[i % len(chromosome_1.jobs_list_1)] not in backup:
                child_1.jobs_list_1[j % len(chromosome_1.jobs_list_1)] = child_2.jobs_list_1[
                    i % len(chromosome_1.jobs_list_1)]
                j += 1

        backup = child_2.jobs_list_1[k:k + size]
        j = k + size

        for i in range(k + size, k + size + len(chromosome_2.jobs_list_1)):
            if child_1.jobs_list_1[i % len(chromosome_2.jobs_list_1)] not in backup:
                child_2.jobs_list_1[j % len(chromosome_2.jobs_list_1)] = child_1.jobs_list_1[
                    i % len(chromosome_2.jobs_list_1)]
                j += 1

        return child_1, child_2