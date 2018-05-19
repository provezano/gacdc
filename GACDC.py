from Utils import Utils
from Population import Population
from UtilsGA import UtilsGA
import random
import numpy as np
from numpy import cumsum
from numpy.random import rand
import matplotlib.pyplot as plt

def weighted_choice(weights, objects):
    cs = cumsum(weights)
    idx = sum(cs < rand())
    if idx >= len(objects): idx = len(objects)-1;
    return idx

def genetic_algorithm(m1, m2, jobs_1, jobs_2, pop_size=100, crossover_rate=0.8, mutation_rate=0.2, generations=100):
    random.seed(a=None)

    t = generations

    population = Population(m1, m2, jobs_1, jobs_2, pop_size)

    population.compute_stats(0)
    #population.print_population()
    #population.print_stats()

    weigth_balance_crossover = 0.0001
    weigth_balance_mutation = 0.0001

    crossover_functions = [UtilsGA.apply_unif_crossover, UtilsGA.apply_pmx_crossover]
    mutation_functions = [UtilsGA.mutation_roll, UtilsGA.mutation_swap, UtilsGA.mutation_slice_random]
    #UtilsGA.mutation_slice_front, UtilsGA.mutation_slice_back

    prob_distribution_mutation = [1.0/len(mutation_functions) for _ in range(len(mutation_functions))]
    prob_distribution_crossover = [1.0/len(crossover_functions) for _ in range(len(crossover_functions))]

    best_individual = population.best_individual_2
    averages_1 = []
    averages_2 = []
    best_evolution_1 = []
    best_evolution_2 = []

    while t > 0:
        print("Generation", generations-t, len(population.population))

        print("---")
        print("Prob distribution Crossover:", prob_distribution_crossover)
        print("Prob distribution Mutation :", prob_distribution_mutation)
        print("---")

        averages_1.append(population.mean_fitness)
        averages_2.append(population.mean_fitness_2)
        best_evolution_1.append(best_individual.makespan_1)
        best_evolution_2.append(best_individual.makespan_2)

        new_population = []

        for i in range(pop_size//2):

            c1, c2 = UtilsGA.tournament_selection(population.population, 3, 2)

            old_c1_makespan_2, old_c2_makespan_2 = c1.makespan_2, c2.makespan_2
            new_c1 = c1
            new_c2 = c2

            if random.random() < crossover_rate:
                c_index = weighted_choice(prob_distribution_crossover, crossover_functions)
                #print(c_index)
                new_c1, new_c2 = crossover_functions[c_index](c1, c2, 1.0)

                new_c1.update_stats()
                new_c2.update_stats()

                if (new_c1.makespan_2 + new_c2.makespan_2) - (old_c1_makespan_2 + old_c2_makespan_2) > 0:
                #if new_c1.makespan_2 > old_c1_makespan_2 or new_c2.makespan_2 >old_c2_makespan_2:
                    for i in range(len(prob_distribution_crossover)):
                        if i == c_index:
                            prob_distribution_crossover[i] += weigth_balance_crossover
                            if prob_distribution_crossover[i] + weigth_balance_crossover > 1.0:
                                prob_distribution_crossover[i] = 0.9
                        else:
                            prob_distribution_crossover[i] -= weigth_balance_crossover / (len(prob_distribution_crossover) - 1)
                            if prob_distribution_crossover[i] < 0.0:
                                prob_distribution_crossover[i] = 0.1

                old_c1_makespan_2, old_c2_makespan_2 = new_c1.makespan_2, new_c2.makespan_2

            if random.random() < mutation_rate:
                m_index = weighted_choice(prob_distribution_mutation, mutation_functions)

                new_c1 = mutation_functions[m_index](new_c1, mutation_rate = 1.0)
                new_c2 = mutation_functions[m_index](new_c2, mutation_rate = 1.0)

                new_c1.update_stats()
                new_c2.update_stats()

                if (new_c1.makespan_2 + new_c2.makespan_2) - (old_c1_makespan_2 + old_c2_makespan_2) > 0:
                #if new_c1.makespan_2 > old_c1_makespan_2 or new_c2.makespan_2 >old_c2_makespan_2:
                    for i in range(len(prob_distribution_mutation)):
                        if i == m_index:
                            prob_distribution_mutation[i] += weigth_balance_mutation
                            if prob_distribution_mutation[i] + weigth_balance_mutation > 1.0:
                                prob_distribution_mutation[i] = 0.8
                        else:
                            prob_distribution_mutation[i] -= weigth_balance_mutation/(len(prob_distribution_mutation)-1)
                            if prob_distribution_mutation[i] < 0.0:
                                prob_distribution_mutation[i] = 0.1

                #new_population.append(new_c1)
                #new_population.append(new_c2)

            new_population.append(new_c1)
            new_population.append(new_c2)

        if population.best_individual_2.makespan_2 < best_individual.makespan_2:
            best_individual = population.best_individual_2

        #if best_individual not in new_population:
        #        new_population.append(best_individual)

        population.population = new_population

        population.update_stats()
        print("Best so far:", best_individual.makespan_2)
        #population.print_population()

        population.compute_stats(generations-t)
        #population.print_stats()

        t -= 1

    return best_individual, averages_1, averages_2, best_evolution_1, best_evolution_2


def main():
    filenames = ["2maquinas/30jobs/40","4maquinas/30jobs/40","10maquinas/30jobs/40"]

    for filename in filenames:
        j1, j2, m1, m2, jobs_1, jobs_2 = Utils.read_from_file(
            "./Intancias/"+filename+".dat")

        for i in range(30):
            best_individual, averages_1, averages_2, best_evolution_1, best_evolution_2 = \
                genetic_algorithm(m1, m2, jobs_1, jobs_2, 100, 0.75, 0.25, 300)

            #print(best_evolution_1)
            with open("results_" + filename.replace("/", "") + ".txt", "a+") as file:
                file.write(str(best_individual.makespan_1) + "\t" + str(best_individual.makespan_2) + "\t" + \
                      str(np.mean(averages_1)) + "\t" + str(np.mean(averages_2)) + "\t" + \
                      str(np.std(averages_1)) + "\t" + str(np.std(averages_2)) + "\t" + \
                      str(np.mean(best_evolution_1)) + "\t" + str(np.mean(best_evolution_2)) + "\t" + \
                      str(np.std(best_evolution_1)) + "\t" + str(np.std(best_evolution_2)) + "\n")

    '''
    plt.plot(averages_1, 'ro')
    plt.ylabel('Média dos makespans do primeiro estágio da população')
    plt.show()

    plt.plot(averages_2, 'ro')
    plt.ylabel('Média dos makespans do segundo estágio da população')
    plt.show()

    plt.plot(best_evolution_1, 'ro')
    plt.ylabel('édia dos makespans do primeiro estágio dos melhores indivíduos')
    plt.show()

    plt.plot(best_evolution_2, 'ro')
    plt.ylabel('Média dos makespans do segundo estágio dos melhores indivíduos')
    plt.show()
    '''


    #print("First stage")
    #result.print_first_stage()
    #print("Second stage")
    #result.print_second_stage()

    
if __name__ == "__main__":
    main()