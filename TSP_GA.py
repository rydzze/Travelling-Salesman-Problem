import os
import time
import json
import random
import operator
import numpy as np
import matplotlib.pyplot as plt

class GA:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix

    def create_new_member(self, num_of_city):
        population = set(np.arange(num_of_city, dtype=int))
        route = list(random.sample(sorted(population), num_of_city))
                
        return route

    def create_starting_population(self, size_of_city, num_of_city):
        population = []
        
        for _ in range(0, size_of_city):
            population.append(self.create_new_member(num_of_city))
            
        return population

    def distance(self, i, j):
        return self.distance_matrix[int(i)][int(j)]

    def fitness(self, route, city_list):
        score = 0

        for i in range(1, len(route)):
            k = int(route[i-1])
            l = int(route[i])
            score = score + self.distance(city_list[k],city_list[l])

        return score

    def crossover(self, a, b):
        child = []
        childA = []
        childB = []
        
        geneA = int(random.random() * len(a))
        geneB = int(random.random() * len(a))
        
        start_gene = min(geneA, geneB)
        end_gene = max(geneA, geneB)
        
        for i in range(start_gene, end_gene):
            childA.append(a[i])
            
        childB = [item for item in b if item not in childA]
        child = childA + childB
        
        return child

    def mutate(self, route, probablity):
        route = np.array(route)

        for swaping_p in range(len(route)):
            if(random.random() < probablity):
                swapedWith = np.random.randint(0, len(route))
                
                temp1 = route[swaping_p]
                temp2 = route[swapedWith]
                route[swapedWith] = temp1
                route[swaping_p] = temp2
        
        return route
        
    def selection(self, population_ranked, elite_size):
        selection_results = []
        result = []
        for i in population_ranked:
            result.append(i[0])
        for i in range(0, elite_size):
            selection_results.append(result[i])
        
        return selection_results

    def rank_routes(self, population, city_list):
        fitness_results = {}
        for i in range(0, len(population)):
            fitness_results[i] = self.fitness(population[i], city_list)

        return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=False)

    def breed_population(self, mating_pool):
        children = []
        for i in range(len(mating_pool)-1):
            children.append(self.crossover(mating_pool[i], mating_pool[i+1]))

        return children

    def mutate_population(self, children, mutation_rate):
        new_generation = []
        for i in children:
            muated_child = self.mutate(i, mutation_rate)
            new_generation.append(muated_child)

        return new_generation

    def mating_pool(self, population, selection_results):
        pool = []
        for i in range(0, len(selection_results)):
            index = selection_results[i]
            pool.append(population[index])

        return pool

    def next_generation(self, city_list, current_population, mutation_rate, elite_size):
        population_rank = self.rank_routes(current_population, city_list)
        selection_result = self.selection(population_rank, elite_size)
        pool = self.mating_pool(current_population, selection_result)
        children = self.breed_population(pool)
        next_generation = self.mutate_population(children, mutation_rate)

        return next_generation

    def genetic_algorithm(self, city_list, show_plot=True, size_population=1000, elite_size=75, mutation_rate=0.01, generation=2000):
        pop = []
        progress = []
        
        num_of_cities = len(city_list)
        
        population = self.create_starting_population(size_population, num_of_cities)
        progress.append(self.rank_routes(population, city_list)[0][1])

        for _ in range(0, generation):
            pop = self.next_generation(city_list, population, mutation_rate, elite_size)
            progress.append(self.rank_routes(pop, city_list)[0][1])
        
        best_route = self.rank_routes(pop, city_list)[0]
        path_taken = pop[best_route[0]].tolist()
        min_cost = str(best_route[1])

        if show_plot:
            plt.plot(progress)
            plt.ylabel("Distance")
            plt.xlabel("Generation")
            plt.show()
        
        return path_taken, min_cost
    

data_dir = 'data/'
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
json_files = sorted(json_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

for file_name in json_files:
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)

    N = data["n_cities"]
    adj = data["distance_matrix"]

    start_time = time.time()
    ga = GA(adj)
    path_taken, min_cost = ga.genetic_algorithm(list(range(N)), show_plot=False)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n=== File: {file_name} ===")
    print("[+] Num. of Cities :", N)
    print("[+] Execution Time : {:.4f} s".format(execution_time))
    print("[+] Path Taken     :", ' '.join(map(str, path_taken)))
    print("[+] Minimum Cost   :", min_cost)