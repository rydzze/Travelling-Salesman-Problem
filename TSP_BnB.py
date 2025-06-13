# This code was taken from GeeksforGeeks website and it is contributed by ng24_7
# https://www.geeksforgeeks.org/traveling-salesman-problem-using-branch-and-bound-2/

import os
import math
import json
import time
maxsize = float('inf')

def copyToFinal(curr_path):
    path_taken[:N + 1] = curr_path[:]
    path_taken[N] = curr_path[0]

def firstMin(adj, i):
    min = maxsize
    for k in range(N):
        if adj[i][k] < min and i != k:
            min = adj[i][k]

    return min

def secondMin(adj, i):
    first, second = maxsize, maxsize
    for j in range(N):
        if i == j:
            continue
        
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]

        elif(adj[i][j] <= second and adj[i][j] != first):
            second = adj[i][j]

    return second

def TSPRec(adj, curr_bound, curr_weight, level, curr_path, visited):
    global min_cost

    if level == N:
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
            if curr_res < min_cost:
                copyToFinal(curr_path)
                min_cost = curr_res
        return

    for i in range(N):
        if (adj[curr_path[level-1]][i] != 0 and visited[i] == False):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]

            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2)

            if curr_bound + curr_weight < min_cost:
                curr_path[level] = i
                visited[i] = True
                
                TSPRec(adj, curr_bound, curr_weight, level + 1, curr_path, visited)

            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp

            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True

def TSP(adj):
    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N

    for i in range(N):
        curr_bound += (firstMin(adj, i) + secondMin(adj, i))
    curr_bound = math.ceil(curr_bound / 2)

    visited[0] = True
    curr_path[0] = 0

    TSPRec(adj, curr_bound, 0, 1, curr_path, visited)


data_dir = 'data/'
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
json_files = sorted(json_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

for file_name in json_files:
    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'r') as f:
        data = json.load(f)

    N = data["n_cities"]
    adj = data["distance_matrix"]

    visited = [False] * N
    path_taken = [None] * (N + 1)
    min_cost = maxsize

    start_time = time.time()
    TSP(adj)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\n=== File: {file_name} ===")
    print("[+] Num. of Cities :", N)
    print("[+] Execution Time : {:.4f} s".format(execution_time))
    print("[+] Path Taken     :", ' '.join(str(city + 1) for city in path_taken))
    print("[+] Minimum Cost   :", min_cost)