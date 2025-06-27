# 🗺📍 Travelling Salesman Problem


## 📌 Introduction  

The **Traveling Salesman Problem (TSP)** is a classic NP-hard combinatorial optimization problem where the goal is to find the **shortest possible route for a salesman to visit a set of cities exactly once and return to the starting city**. 

This project provides a comprehensive analysis of four algorithmic approaches to solve TSP: **Branch and Bound (B&B)**, **Ant Colony Optimization (ACO)**, **Evolutionary Algorithm (EA)**, and **Simulated Annealing (SA)**.


## 🧮 Algorithms Applied  

### 🌲 Branch and Bound (B&B)

Developed by **A. H. Land and A. G. Doig (1960)**, B&B uses a tree-search strategy to explore partial tours while pruning branches that exceed a calculated lower bound. It guarantees optimal solutions but suffers from factorial time complexity. Code sourced from [GeeksforGeeks (ng24_7)](https://www.geeksforgeeks.org/traveling-salesman-problem-using-branch-and-bound-2/).  

### 🐜 Ant Colony Optimization (ACO)  

Introduced by **Marco Dorigo (1992)**, ACO simulates ant foraging behavior using pheromone trails to probabilistically construct routes. It balances exploration and exploitation for near-optimal solutions. Code adapted from [Vampboy's GitHub](https://github.com/Vampboy/Ant-Colony-Optimization) and [tribasuki74's Medium article](https://medium.com/thelorry-product-tech-data/ant-colony-optimization-to-solve-the-travelling-salesman-problem-d19bd866546e).  

### 🧬 Evolutionary Algorithm (EA)  

Attributed to **John Holland**, EA mimics natural selection through genetic operations (selection, crossover, mutation) to evolve populations of candidate solutions. Code implemented from [avitomar12's GitHub](https://github.com/avitomar12/TSP-using-Genetic-Algorithm) and [accompanying Medium article](https://medium.com/thecyphy/travelling-salesman-problem-using-genetic-algorithm-130ab957f165).  

### 🌡️ Simulated Annealing (SA)  

Invented by **Kirkpatrick, Gelatt, and Vecchi**, SA is inspired by metallurgical annealing. It uses probabilistic acceptance of worse solutions at high "temperatures" to escape local optima. Code sourced from [Paolo Lapo Cerni's GitHub](https://github.com/paololapo/Simulated_annealing_for_TSP).  


## 📊 Experimental Analysis  

Experiments tested all algorithms on TSP instances ranging from **5 to 50 cities**:  

| Algorithm          | 5 Cities | 25 Cities | 50 Cities        | Scalability      | Solution Quality |  
|--------------------|----------|-----------|------------------|------------------|------------------|  
| **B&B**            | 0.0001s  | 4846s     | ∞                | Poor (>20 cities)| Optimal          |  
| **ACO**            | 0.19s    | 1.41s     | 3.59s            | Excellent        | Near-optimal     |  
| **EA**             | 5.53s    | 24.07s    | 47.19s           | Good             | Near-optimal     |  
| **SA**             | 0.09s    | 0.12s     | 0.19s            | Best (≤40 cities)| Near-optimal     |  

### Key Findings:  

- **B&B** is only feasible for small instances (<20 cities) due to factorial explosion.  
- **ACO** offers the best balance: linear time scaling and high solution quality.  
- **EA** provides consistent results but is slower due to generational evolution.  
- **SA** is fastest for small-to-medium instances but struggles slightly beyond 40 cities.  


## 🏆 Contribution

We would like to thank the following team members for their contributions:

- [Muhammad Ariff Ridzlan](https://github.com/rydzze)
- [Noor Alia Alisa](https://github.com/alia4lisa)
- [Low Wei Jie](https://github.com/yumiian)
- [Muhammad Nabil Irfan](https://github.com/nabilang)
