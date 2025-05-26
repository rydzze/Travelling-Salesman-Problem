import os
import random
import numpy as np
from typing import Tuple

class TSPGenerator:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_from_coordinates(self, n_cities: int, coordinate_range: int = 100) -> Tuple[np.ndarray, int]:
        cities = []
        for _ in range(n_cities):
            x = random.randint(0, coordinate_range)
            y = random.randint(0, coordinate_range)
            cities.append((x, y))
        
        distance_matrix = np.zeros((n_cities, n_cities), dtype=int)
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    x1, y1 = cities[i]
                    x2, y2 = cities[j]

                    distance = int(round(np.sqrt((x1 - x2)**2 + (y1 - y2)**2)))
                    distance_matrix[i][j] = distance
        
        return n_cities, distance_matrix

def create_coordinate_tsp(n_cities: int, coord_range: int = 100, seed=None):
    tsp = TSPGenerator(seed)
    
    return tsp.generate_from_coordinates(n_cities, coord_range)

def export_to_json(n_cities: int, distance_matrix: np.ndarray, filename: str, output_dir: str = "data"):
    distance_matrix_list = distance_matrix.tolist()
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write('{\n')
        f.write(f'\t"n_cities": {n_cities},\n')
        f.write('\t"distance_matrix": [\n')
        
        for i, row in enumerate(distance_matrix_list):
            row_str = '[' + ', '.join(f'{val:3d}' for val in row) + ']'

            if i < len(distance_matrix_list) - 1:
                f.write(f'\t\t{row_str},\n')
            else:
                f.write(f'\t\t{row_str}\n')
        
        f.write('\t]\n')
        f.write('}\n')
    
    print(f"[+] TSP data exported to {filepath}")

if __name__ == "__main__":
    output_dir = "data"
    total_files = 0

    print("\n=== Multiple TSP Data Generator ===")
    for i in range(5, 51, 5):
        print(f"\n[*] Generating TSP with {i} cities ...")
        n, distances = create_coordinate_tsp(i, coord_range=100, seed=None)
        export_to_json(n, distances, f"tsp_{i}.json", output_dir=output_dir)
        total_files += 1
    
    print(f"\n[+] Generated {total_files} TSP data successfully!\n")
    


    # print("\n=== Verification ===\n")
    # print(f"Matrix shape: {distances.shape}")
    # print(f"Data type: {distances.dtype}")
    # print(f"Diagonal sum (should be 0): {np.trace(distances)}")
    # print(f"Is symmetric: {np.allclose(distances, distances.T)}\n")