import sys, getopt

import copy
import matplotlib.pyplot as plt
import numpy as np
import random
import time

import scipy as sp
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    #return m, m - h, m + h
    return h


from scipy.spatial.distance import pdist

from Population import Population
from Utils import Utils
from UtilsAE import UtilsAE

ordering_second_stage_options = ['LPT', 'SRT', 'LNS', 'SNS', 'AVA1', 'AVA2', 'Random', 'Random', 'Random', 'Random',
                                 'Random', 'Random']
convert_index = lambda i, j, n: n * j - j * (j + 1) / 2 + i - 1 - j


def isNonIncreasing(l):
    return sorted(l, reverse=True) == l


def weighted_choice(weights, objects):
    cs = np.cumsum(weights)
    idx = sum(cs < np.random.rand())
    return idx


def evolutionary_algorithm(m1, m2, jobs_1, jobs_2, pop_size=100, crossover_rate=0.8, mutation_rate=0.2, generations=100, total_selection=2):
    random.seed(a=None)

    population = Population(m1, m2, jobs_1, jobs_2, pop_size)

    crossover_functions = [UtilsAE.apply_ox_crossover, UtilsAE.apply_pmx_crossover]
    mutation_functions = [UtilsAE.mutation_slice_front, UtilsAE.insertion_mutation, UtilsAE.inversion_mutation,
                          UtilsAE.mutation_slice_random, UtilsAE.mutation_partition]#, UtilsAE.mutation_roll, UtilsAE.mutation_slice_back]

    best_mutation = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    prob_distribution_mutation = [v/sum(best_mutation.values()) for _,v in best_mutation.items()]

    best_crossover = {0: 1, 1: 1}
    prob_distribution_crossover = [v / sum(best_crossover.values()) for _, v in best_crossover.items()]

    best_chromosome = copy.deepcopy(population.best_individual_2)

    pop_mean_list = []


    # print("Mean dist:", np.mean(dist))
    # print("Min dist:", np.min(dist))
    # print("Max dist:", np.max(dist))
    # print("Std dev dist:", np.std(dist))

    for g in range(0, generations):
        print("Generation:", g)

        print("--- Population Stats ---")
        population.compute_stats(g)
        print("Mean:", population.mean_fitness_2)
        print("Std. Dev.:", population.std_fitness_2)
        print("Best Makespan:", best_chromosome.makespan_2)
        print("PD Mutation:", prob_distribution_mutation)
        print("PD Mutation:", prob_distribution_crossover)
        print("---")

        #population.print_population()
        #input()

        pop_mean_list.append(population.mean_fitness_2)

        changed = False

        #selected_chromosomes = UtilsAE.uniform_selection(population.population, total_selection)
        selected_chromosomes = UtilsAE.tournament_selection(population.population, total_chromosomes_tournament=7, total_chromosomes=total_selection)


        if random.random() < crossover_rate:
            changed = True

            for i in range(len(selected_chromosomes)):
                c_index = weighted_choice(prob_distribution_crossover, crossover_functions)
                for j in range(len(selected_chromosomes)):
                    if i != j:
                        c1, c2 = crossover_functions[c_index](selected_chromosomes[i], selected_chromosomes[j])
                        c1.update_stats()
                        c2.update_stats()

                        best_c = c1 if c1.makespan_2 > c2.makespan_2 else c2

                        if selected_chromosomes[i].makespan_2 - best_c.makespan_2 > 0:
                            best_crossover[c_index] += 1
                        elif selected_chromosomes[i].makespan_2 - best_c.makespan_2 < 0:
                            if best_crossover[c_index] != 1:
                                best_crossover[c_index] -= 1

                        selected_chromosomes[i] = best_c
                        if best_c.makespan_2 < best_chromosome.makespan_2:
                            best_chromosome = copy.deepcopy(best_c)
                            best_chromosome.iter = g

                prob_distribution_crossover = [v / sum(best_crossover.values()) for _, v in best_crossover.items()]



        if random.random() < mutation_rate:
            changed = True
            for i in range(len(selected_chromosomes)):
                m_index = weighted_choice(prob_distribution_mutation, mutation_functions)

                c1 = mutation_functions[m_index](selected_chromosomes[i])
                c1.update_stats()

                if selected_chromosomes[i].makespan_2 - c1.makespan_2 > 0:
                    best_mutation[m_index] += 1
                elif selected_chromosomes[i].makespan_2 - c1.makespan_2 < 0:
                    if best_mutation[m_index] != 1:
                        best_mutation[m_index] -= 1

                selected_chromosomes[i] = c1

                prob_distribution_mutation = [v / sum(best_mutation.values()) for _, v in best_mutation.items()]

        #selected_chromosomes_die = UtilsAE.roulette_wheel_selection_survive(population.population, total_selection)
        #selected_chromosomes_die = UtilsAE.uniform_selection(range(len(population.population)), total_selection)
        if changed:

            selected_chromosomes_die = UtilsAE.tournament_selection_survive(population.population,
                                                                          total_chromosomes_tournament=5,
                                                                           total_chromosomes=total_selection)

            #selected_chromosomes_die = UtilsAE.uniform_selection(range(len(population.population)), total_selection)
            for i in range(total_selection):
                selected_chromosomes[i].id = population.population[selected_chromosomes_die[i]].id
                population.population[selected_chromosomes_die[i]] = selected_chromosomes[i]

            if population.best_individual_2.makespan_2 < best_chromosome.makespan_2:
                best_chromosome = copy.deepcopy(population.population[population.best_individual_index_2])
                best_chromosome.iter = g

    """
    print("Mean dist:", np.mean(dist), mean)
    print("Min dist:", np.min(dist))
    print("Max dist:", np.max(dist))
    print("Std dev dist:", np.std(dist))

    x = ra nge(len(pop_mean_list))
    plt.yscale('linear')
    plt.plot(x, pop_mean_list, 'o--')
    plt.title("Makespan average among generations")
    plt.show()

    x = range(len(dist_mean_list))
    plt.yscale('linear')
    plt.loglog(x, dist_mean_list, 'o--')
    plt.title("Makespan average among generations")
    plt.show()
    """

    return best_chromosome, np.mean(pop_mean_list), np.std(pop_mean_list)


def main(argv):
    maquinas = ["[2,4]maquinas", "[2,10]maquinas", "2maquinas", "4maquinas", "10maquinas"]
    jobs = [str(n) + "jobs" for n in range(20,90,10)]
    seed = [str(n) for n in range(25, 317, 1)]

    #maquinas = ["[2,4]maquinas"]
    #jobs = ["80jobs"]
    #seed = ["17"]


    # Parameters
    pop_size = 100
    generations = 1000
    total_experiments = 1
    crossover_rate = 0.8
    mutation_rate = 0.2
    total_select = 2
    filename = ""

    try:
        opts, args = getopt.getopt(argv, "hcr:mr:", ["crossover_rate=", "mutation_rate="])
    except getopt.GetoptError:
        print('GACDC.py -cr <crossover rate> -mr <mutation rate>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('GACDC.py -cr <crossover rate> -mr <mutation rate>')
            sys.exit()
        elif opt in ("-cr", "--mutation_rate"):
            crossover_rate = arg
        elif opt in ("-mr", "--crossover_rate"):
            mutation_rate = arg

    for m in maquinas:
        with open("./results/general/results_" + m + "_.txt", "a+") as file:
            file.write("GACDC - Population size: " + str(pop_size) + " - Generations: " + str(generations) + "\n")
            file.write("Instance\tBest\tAverage\tStd. Dev.\tConf. Interval\tIter. Best\tAvg. Iter.\tIter. Std Dev\tAvg. Exec. Time\tStd. Dev. Exec. Time\n")

    for j in jobs:
        for s in seed:
            for m in maquinas:
                filename = m + "/" + j + "/" + s

                j1, j2, m1, m2, jobs_1, jobs_2 = Utils.read_from_file(
                    "/Users/coutinho/PycharmProjects/GACD/Dados/" + filename + ".dat")

                best_individual_list_1 = []
                best_individual_list_2 = []
                iteration_list = []

                pop_mean_list = []
                pop_std_list = []
                execution_times = []

                with open("./results/individual/results_" + filename.replace("/", "") + ".txt", "a+") as file:
                    file.write("GACDC - Population size: " + str(pop_size) + " - Generations: " + str(generations) + "\n")
                    file.write("ID\tMakespan(1)\tMakespan(2)\tAverage\tStd. Dev.\n")

                for exp_num in range(total_experiments):
                    start_time = time.time()
                    best_individual, pop_mean, pop_std = evolutionary_algorithm(m1, m2, jobs_1, jobs_2, pop_size, crossover_rate, mutation_rate, generations, total_select)
                    execution_time = time.time() - start_time

                    best_individual_list_1.append(best_individual.makespan_1)
                    best_individual_list_2.append(best_individual.makespan_2)
                    iteration_list.append(best_individual.iter)
                    execution_times.append(execution_time)
                    pop_mean_list.append(pop_mean)
                    pop_std_list.append(pop_std)

                    with open("./results/individual/results_" + filename.replace("/", "") + ".txt", "a+") as file:
                        file.write(str(exp_num+1) + "\t"+ str(best_individual.makespan_1) + "\t" + str(best_individual.makespan_2) + "\t" + \
                                   str(pop_mean) + "\t" + str(pop_std) + "\t" +str(execution_time) + "\n")

                with open("./results/individual/results_" + filename.replace("/", "") + ".txt", "a+") as file:
                    file.write(
                        "\nAverage\t" + str(np.mean(best_individual_list_1)) + "\t" + str(np.mean(best_individual_list_2)) + "\t" + \
                        str(np.mean(pop_mean_list)) + "\t" + str(np.mean(pop_std_list)) + str(np.mean(execution_times)) + "\nStd. Dev.\t" + \
                        str(np.std(best_individual_list_1)) + "\t" + str(np.std(best_individual_list_2)) + "\t" + \
                        str(np.std(pop_mean_list)) + "\t" + str(np.std(pop_std_list)) + str(np.std(execution_times)) + "\nMin\t" + \
                        str(np.min(best_individual_list_1)) + "\t" + str(np.min(best_individual_list_2)) + "\t" + \
                        str(np.min(pop_mean_list)) + "\t" + str(np.min(pop_std_list)) + str(np.max(execution_times)) + "\nMax\t" + \
                        str(np.max(best_individual_list_1)) + "\t" + str(np.max(best_individual_list_2)) + "\t" + \
                        str(np.max(pop_mean_list)) + "\t" + str(np.max(pop_std_list)) + str(np.min(execution_times)) + "\nConf. Interval\t" + \
                        str(mean_confidence_interval(best_individual_list_1)) + "\t" +
                        str(mean_confidence_interval(best_individual_list_2)) + "\t"+
                        str(mean_confidence_interval(pop_mean_list)) + "\t" + str(mean_confidence_interval(pop_std_list)) + \
                        str(mean_confidence_interval(execution_times)) + "\n")

                with open("./results/general/results_" + m + "_.txt", "a+") as file:
                    file.write(filename.replace("/", "") + "\t" + str(np.min(best_individual_list_2)) + "\t" +
                               str(np.mean(best_individual_list_2)) + "\t" + str(np.std(best_individual_list_2)) + "\t" +
                               str(mean_confidence_interval(best_individual_list_2)) + "\t" +
                               str(iteration_list[best_individual_list_2.index(np.min(best_individual_list_2))]) + "\t" +
                               str(np.mean(iteration_list)) + "\t" +
                               str(np.std(iteration_list)) + "\t" +
                               str(np.mean(execution_times))+ "\t" +
                               str(np.mean(execution_times))+ "\n")

if __name__ == "__main__":
    main(sys.argv[1:])