import numpy as np
import json
import time
import os
from glob import glob

# ----------- SUPPORT FUNCTIONS -----------

def L2(cities, i, j):
    dx = cities["x"][i] - cities["x"][j]
    dy = cities["y"][i] - cities["y"][j]
    return (dx**2 + dy**2)**0.5

def D_matrix(distance, cities):
    N = len(cities["x"])
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i][j] = distance(cities, i, j)
    return D

def L(path, D):
    return sum(D[path[k+1]][path[k]] for k in range(len(path)-1))

def transpose(arr):
    temp = arr.copy()
    idx = np.random.choice(np.arange(len(temp))[1:-1], size=2, replace=False)
    temp[idx[0]], temp[idx[1]] = temp[idx[1]], temp[idx[0]]
    return temp

# ----------- SIMULATED ANNEALING LIGHT -----------

def SA_light(D, T0, T_f, alpha):
    idx = np.arange(D.shape[1])
    conf_i = np.random.permutation(idx)
    conf_i = np.append(conf_i, conf_i[0])
    L_i = L(conf_i, D)

    best = L_i
    T = T0

    while T > T_f:
        conf_t = transpose(conf_i)
        L_t = L(conf_t, D)

        if L_t < L_i:
            conf_i = conf_t
            L_i = L_t
            best = min(L_t, best)
        else:
            best = min(L_i, best)
            if np.exp(-(L_t - L_i) / T) > np.random.uniform():
                conf_i = conf_t
                L_i = L_t

        T = alpha * T

    return conf_i, best

# ----------- SA RUNNER FOR JSON FILES -----------

def run_sa_on_json(file_path, T0=1000, T_f=1e-3, alpha=0.995):
    with open(file_path, 'r') as f:
        data = json.load(f)

    if "distance_matrix" in data:
        D = np.array(data["distance_matrix"])
    elif "cities" in data:
        coords = { "x": [c["x"] for c in data["cities"]], "y": [c["y"] for c in data["cities"]] }
        D = D_matrix(L2, coords)
    else:
        raise ValueError(f"{file_path} does not contain required keys.")

    start = time.time()
    final_path, min_cost = SA_light(D, T0, T_f, alpha)
    exec_time = time.time() - start

    file_name = os.path.basename(file_path)
    N = D.shape[0]

    print(f"\n=== File: {file_name} ===")
    print("[+] Num. of Cities :", N)
    print("[+] Execution Time : {:.4f} s".format(exec_time))
    print("[+] Path Taken     :", ' '.join(str(city + 1) for city in final_path))
    print("[+] Minimum Cost   :", round(min_cost, 3))

# ----------- MAIN LOOP -----------

folder_path = r"C:\Users\12213\OneDrive\Desktop\TSP.SA\Data"
json_files = sorted(glob(os.path.join(folder_path, "*.json")))

for file_path in json_files:
    run_sa_on_json(file_path)
