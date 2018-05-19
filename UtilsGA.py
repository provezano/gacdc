import copy
import numpy as np
import random
from Population import Population
from Chromosome import Chromosome


class UtilsGA:

    def uniform_selection(population, total_chromosomes=3):
        return random.sample(population, total_chromosomes)

    def tournament_selection(population, total_chromosomes_tournament=3, total_chromosomes=1):
        chromosomes_selected = set()

        while len(chromosomes_selected) < total_chromosomes:
            tournament_pop = UtilsGA.uniform_selection(population, total_chromosomes_tournament)

            min_index = np.argmin([c.makespan_2 for c in tournament_pop])
            chromosomes_selected.add(tournament_pop[min_index])
            #print(len(chromosomes_selected), chromosomes_selected)

        return list(chromosomes_selected)

    def tournament_selection_survive(population, total_chromosomes_tournament=3, total_chromosomes=1):
        chromosomes_selected = set()

        while len(chromosomes_selected) < total_chromosomes:
            tournament_pop = UtilsGA.uniform_selection(population, total_chromosomes_tournament)
            min_index = np.argmax([c.makespan_2 for c in tournament_pop])
            chromosomes_selected.add(tournament_pop[min_index])

        return list(chromosomes_selected)

    def mutation_swap(chromosome, mutation_rate=0.1):
        new_chromosome = copy.deepcopy(chromosome)

        for i in range(len(new_chromosome.jobs_list_1)):
            if random.random() <= 0.2:
                k = random.choice(range(len(new_chromosome.jobs_list_1)))
                new_chromosome.jobs_list_1[i], new_chromosome.jobs_list_1[k] = new_chromosome.jobs_list_1[k], new_chromosome.jobs_list_1[i]

        for i in range(len(new_chromosome.jobs_list_2)):
            if random.random() <= 0.2:
                k = random.choice(range(len(new_chromosome.jobs_list_2)))
                new_chromosome.jobs_list_2[i], new_chromosome.jobs_list_2[k] = new_chromosome.jobs_list_2[k], new_chromosome.jobs_list_2[i]

        return new_chromosome

    def mutation_roll(chromosome, mutation_rate=0.3):
        new_chromosome = copy.deepcopy(chromosome)

        if random.random() < mutation_rate:
            new_chromosome.jobs_list_1 = np.roll(new_chromosome.jobs_list_1, random.randint(1,len(new_chromosome.jobs_list_1)-1))
            new_chromosome.jobs_list_1 =new_chromosome.jobs_list_1.tolist()

        if random.random() < mutation_rate:
            new_chromosome.jobs_list_2 = np.roll(new_chromosome.jobs_list_2, random.randint(1,len(new_chromosome.jobs_list_2)-1))
            new_chromosome.jobs_list_2 =new_chromosome.jobs_list_2.tolist()

        return new_chromosome

    def mutation_slice_front(chromosome, mutation_rate=0.3):
        #assert size < len(chromosome.jobs_list_1), "size must be < length(chromosome)"
        size = random.randint(1, len(chromosome.jobs_list_1)//1)

        new_chromosome = copy.deepcopy(chromosome)

        if random.random() < mutation_rate:
            k = random.choice(range(len(chromosome.jobs_list_1)-size))
            new_chromosome.jobs_list_1 = chromosome.jobs_list_1[k:k+size]
            new_chromosome.jobs_list_1.extend(chromosome.jobs_list_1[0:k])
            new_chromosome.jobs_list_1.extend(chromosome.jobs_list_1[k+size:])

        size = random.randint(1, len(chromosome.jobs_list_2)//2)
        if random.random() < mutation_rate:
            k = random.choice(range(len(chromosome.jobs_list_2)-size))
            new_chromosome.jobs_list_2 = chromosome.jobs_list_2[k:k+size]
            new_chromosome.jobs_list_2.extend(chromosome.jobs_list_2[0:k])
            new_chromosome.jobs_list_2.extend(chromosome.jobs_list_2[k+size:])

        return new_chromosome


    def mutation_slice_back(chromosome, mutation_rate=0.3):
        #assert size < len(chromosome.jobs_list_1), "size must be < length(chromosome)"
        size = random.randint(1, len(chromosome.jobs_list_1)//2)

        new_chromosome = copy.deepcopy(chromosome)

        if random.random() < mutation_rate:
            k = random.choice(range(len(chromosome.jobs_list_1)-size))
            new_chromosome.jobs_list_1=(chromosome.jobs_list_1[0:k])
            new_chromosome.jobs_list_1.extend(chromosome.jobs_list_1[k+size:])
            new_chromosome.jobs_list_1.extend(chromosome.jobs_list_1[k:k + size])

        size = random.randint(1, len(chromosome.jobs_list_2)//2)
        if random.random() < mutation_rate:
            k = random.choice(range(len(chromosome.jobs_list_2)-size))
            new_chromosome.jobs_list_2 = (chromosome.jobs_list_2[0:k])
            new_chromosome.jobs_list_2.extend(chromosome.jobs_list_2[k+size:])
            new_chromosome.jobs_list_2.extend(chromosome.jobs_list_2[k:k + size])

        return new_chromosome


    def mutation_slice_random(chromosome, mutation_rate=0.3):
        #assert size < len(chromosome.jobs_list_1), "size must be < length(chromosome)"
        size = random.randint(1, len(chromosome.jobs_list_1)//2)

        new_chromosome = copy.deepcopy(chromosome)

        if random.random() < mutation_rate:
            k = random.choice(range(len(new_chromosome.jobs_list_1)-size))

            backup = new_chromosome.jobs_list_1[k:k + size]
            del new_chromosome.jobs_list_1[k:k + size]
            pos = random.randint(0, len(new_chromosome.jobs_list_1))

            new_chromosome.jobs_list_1[pos:pos] = backup

        size = random.randint(1, len(new_chromosome.jobs_list_2)//2)
        if random.random() < mutation_rate:
            k = random.choice(range(len(new_chromosome.jobs_list_2)-size))

            backup = new_chromosome.jobs_list_2[k:k + size]
            del new_chromosome.jobs_list_2[k:k + size]
            pos = random.randint(0, len(new_chromosome.jobs_list_2))

            new_chromosome.jobs_list_2[pos:pos] = backup

        return new_chromosome

    def apply_pmx_crossover(chromosome_1, chromosome_2, crossover_rate):
        child_1, child_2 = copy.deepcopy(chromosome_1), copy.deepcopy(chromosome_2)

        if random.random() < crossover_rate:
            size = random.randint(1, len(chromosome_2.jobs_list_1)//2)
            k = random.randint(0, len(chromosome_2.jobs_list_1)-size)

            new_child_1 = [None for i in range(len(chromosome_1.jobs_list_1))]

            for i in range(k, k+size):
                new_child_1[i] = child_1.jobs_list_1[i]

            for i in range(k, k+size):
                if child_2.jobs_list_1[i] not in [j for j in new_child_1 if j is not None]:

                    ind = child_2.jobs_list_1.index(child_1.jobs_list_1[i])
                    while k <= ind < k+size:
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

        if random.random() < crossover_rate:
            size = random.randint(1, len(chromosome_2.jobs_list_2)//2)
            k = random.randint(0, len(chromosome_2.jobs_list_2)-size)

            new_child_1 = [None for i in range(len(chromosome_1.jobs_list_2))]

            for i in range(k, k+size):
                new_child_1[i] = child_1.jobs_list_2[i]

            for i in range(k, k+size):
                if child_2.jobs_list_2[i] not in [j for j in new_child_1 if j is not None]:

                    ind = child_2.jobs_list_2.index(child_1.jobs_list_2[i])
                    while k <= ind < k+size:
                        ind = child_2.jobs_list_2.index(child_1.jobs_list_2[ind])

                    new_child_1[ind] = child_2.jobs_list_2[i]

            for i in range(len(chromosome_1.jobs_list_2)):
                if new_child_1[i] is None:
                    new_child_1[i] = child_2.jobs_list_2[i]

            new_child_2 = [None for i in range(len(chromosome_2.jobs_list_2))]

            for i in range(k, k + size):
                new_child_2[i] = child_2.jobs_list_2[i]

            for i in range(k, k + size):
                if child_1.jobs_list_2[i] not in [j for j in new_child_2 if j is not None]:

                    ind = child_1.jobs_list_2.index(child_2.jobs_list_2[i])
                    while k <= ind < k + size:
                        ind = child_1.jobs_list_2.index(child_2.jobs_list_2[ind])

                    new_child_2[ind] = child_1.jobs_list_2[i]

            for i in range(len(chromosome_2.jobs_list_2)):
                if new_child_2[i] is None:
                    new_child_2[i] = child_1.jobs_list_2[i]

            child_1.jobs_list_2 = new_child_1[:]
            child_2.jobs_list_2 = new_child_2[:]

        return child_1, child_2

    def apply_unif_crossover(chromosome_1, chromosome_2, crossover_rate):
        child_1, child_2 = copy.deepcopy(chromosome_1), copy.deepcopy(chromosome_2)

        if random.random() < crossover_rate:
            size = random.randint(1, len(chromosome_2.jobs_list_1)//2)
            k = random.randint(0, len(chromosome_2.jobs_list_1)-size)

            backup = child_1.jobs_list_1[k:k+size]

            j = k+size
            for i in range(k+size, k+size+len(chromosome_1.jobs_list_1)):
                if child_2.jobs_list_1[i % len(chromosome_1.jobs_list_1)] not in backup:
                    child_1.jobs_list_1[j % len(chromosome_1.jobs_list_1)] = child_2.jobs_list_1[i % len(chromosome_1.jobs_list_1)]
                    j+=1

            backup = child_2.jobs_list_1[k:k+size]
            j = k+size

            for i in range(k+size, k+size+len(chromosome_2.jobs_list_1)):
                if child_1.jobs_list_1[i % len(chromosome_2.jobs_list_1)] not in backup:
                    child_2.jobs_list_1[j % len(chromosome_2.jobs_list_1)] = child_1.jobs_list_1[i % len(chromosome_2.jobs_list_1)]
                    j+=1

        if random.random() < crossover_rate:
            size = random.randint(1, len(chromosome_1.jobs_list_2)//2)
            k = random.randint(0, len(chromosome_1.jobs_list_2)-size)

            backup = child_1.jobs_list_2[k:k+size]

            j = k+size
            for i in range(k+size, k+size+len(chromosome_1.jobs_list_2)):
                if child_2.jobs_list_2[i % len(chromosome_1.jobs_list_2)] not in backup:
                    child_1.jobs_list_2[j % len(chromosome_1.jobs_list_2)] = child_2.jobs_list_2[i % len(chromosome_1.jobs_list_2)]
                    j += 1

            backup = child_2.jobs_list_2[k:k+size]
            j = k+size

            for i in range(k+size, k+size+len(chromosome_2.jobs_list_2)):
                if child_1.jobs_list_2[i % len(chromosome_2.jobs_list_2)] not in backup:
                    child_2.jobs_list_2[j % len(chromosome_2.jobs_list_2)] = child_1.jobs_list_2[i % len(chromosome_2.jobs_list_2)]
                    j += 1

        return child_1, child_2


"""
    def roulette_wheel_selection(population, total_chromosomes=1):
        fitnesses = [c.makespan_2 for c in population]
        total_fitness = float(sum(fitnesses))
        rel_fitness = [f/total_fitness for f in fitnesses]

        # Generate probability intervals for each individual
        probs = ([(sum(rel_fitness[:i + 1])) for i in range(len(rel_fitness))])
        #print(probs)
        #probs.reverse() #Smaller the makespan is the better
        #print(probs)

        new_population = []
        for n in range(total_chromosomes):
            r = random.random()
            print(r)
            for (i, individual) in enumerate(population):
                if r <= probs[i]:
                    new_population.append(individual)
                    break

        print(probs)
        return new_population
"""