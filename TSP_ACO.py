# This code was taken from GitHub authored by Vampboy and from Medium website contributed by tribasuki74
# https://github.com/Vampboy/Ant-Colony-Optimization
# https://medium.com/thelorry-product-tech-data/ant-colony-optimization-to-solve-the-travelling-salesman-problem-d19bd866546e

import os
import json
import time
import numpy as np
import random

# ===== ACO Parameters Settings =====
np.random.seed(42)                              # Ensures reproducibility for numpy randomness
num_ants = 10                                   # Number of ants in each iteration
num_iterations = 100                            # Number of ACO iterations
alpha = 1                                       # Pheromone trail
beta = 2                                        # Visibility (heuristic) (1/distance))
evaporation_rate = 0.5

# ===== ACO Algorithm =====
# ACO function
def ACO(distance_matrix):
    num_cities = len(distance_matrix)

    # Initialize pheromone matrix (τ) and visibility matrix (η = 1/distance)
    pheromone = np.ones((num_cities, num_cities))
    visibility = 1 / np.where(distance_matrix == 0, np.inf, distance_matrix)    # Avoid division by 0
    np.fill_diagonal(visibility, 0)                                             # No self-loop visibility

    best_route = None
    best_cost = float('inf')
    costHistory = []

    for iteration in range(num_iterations):
        # Stores complete paths of all ants
        all_routes = np.ones((num_ants, num_cities + 1), dtype=int) * -1

        # Randomly assign a starting city to each ant
        start_cities = np.random.choice(num_cities, size=num_ants, replace=True)

        for k in range(num_ants):
            start_city = start_cities[k]
            all_routes[k, 0] = start_city
            visited = set([start_city])

            # Construct a tour by choosing next cities based on probabilities
            for step in range(1, num_cities):
                current_city = all_routes[k, step - 1]
                probabilities = np.zeros(num_cities)

                # Compute transition probabilities for each unvisited city
                for j in range(num_cities):
                    if j not in visited:
                        tau = pheromone[current_city][j] ** alpha
                        eta = visibility[current_city][j] ** beta
                        probabilities[j] = tau * eta

                prob_sum = np.sum(probabilities)
                if prob_sum == 0:
                    options = [j for j in range(num_cities) if j not in visited]
                    next_city = random.choice(options)
                else:
                    probabilities /= prob_sum
                    next_city = np.random.choice(range(num_cities), p=probabilities)

                all_routes[k, step] = next_city
                visited.add(next_city)

            # Complete the loop by returning to the starting city
            all_routes[k, -1] = start_city  # Return to start city

        # Compute tour lengths for all ants
        distances = np.array([calculate_distance(route[:-1], distance_matrix) for route in all_routes])
        min_index = np.argmin(distances)
        min_cost = distances[min_index]

        if min_cost < best_cost:
            best_cost = min_cost
            best_route = all_routes[min_index]

        costHistory.append(best_cost)

        # Evaporation
        pheromone *= (1 - evaporation_rate)

        # Deposit
        for k in range(num_ants):
            for i in range(num_cities):
                from_city = all_routes[k][i]
                to_city = all_routes[k][i + 1]
                total_deposit = 1.0 / distances[k]
                pheromone[from_city][to_city] += total_deposit
                pheromone[to_city][from_city] += total_deposit

    return best_route.tolist(), int(best_cost), costHistory

    # Calculate route cost

def calculate_distance(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route)-1)) + distance_matrix[route[-1]][route[0]]

# ===== Main Function for all JSON Files =====
if __name__ == "__main__":
    data_dir = 'data/'
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    json_files = sorted(json_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    for file_name in json_files:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r') as f:
            data = json.load(f)

        N = data["n_cities"]
        adj = data["distance_matrix"]
        distance_matrix = np.array(adj)

        start_time = time.time()
        path_taken, min_cost, _ = ACO(distance_matrix)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"\n=== File: {file_name} ===")
        print("[+] Num. of Cities :", N)
        print("[+] Execution Time : {:.4f} s".format(execution_time))
        print("[+] Path Taken     :", ' '.join(str(city + 1) for city in path_taken))
        print("[+] Minimum Cost   :", min_cost)