# Metaheuristic Algorithms (MAs) Evaluated

The following 20 MAs were chosen for their diversity, historical significance, and potential for innovative application:

1. Genetic Algorithm (GA)
2. Particle Swarm Optimization (PSO)
3. Ant Colony Optimization (ACO)
4. Simulated Annealing (SA)
5. Tabu Search (TS)
6. Differential Evolution (DE)
7. Evolutionary Strategies (ES)
8. Grey Wolf Optimizer (GWO)
9. Firefly Algorithm (FA)
10. Whale Optimization Algorithm (WOA)
11. Bat Algorithm (BA)
12. Cuckoo Search (CS)
13. Dragonfly Algorithm (DA)
14. Flower Pollination Algorithm (FPA)
15. Harmony Search (HS)
16. Moth-Flame Optimization (MFO)
17. Salp Swarm Algorithm (SSA)
18. Grasshopper Optimization Algorithm (GOA)
19. Sine Cosine Algorithm (SCA)
20. Biogeography-Based Optimizer (BBO)

## Why These MAs?

These algorithms were selected for their unique approaches to exploring and exploiting the search space, their proven track records in various optimization tasks, and their potential adaptability to the domain of medical image analysis.

## Selection Process

### Benchmark Functions

The evaluation of MAs involved testing each algorithm against a set of benchmark functions known for their challenging optimization landscapes. These functions include:
#### Ackley's Function !!
A popular optimization benchmark function, Ackley's function is known for its complex landscape with many local minima. It is designed to test an algorithm's ability to escape local minima and find the global minimum.
![Ackley's Function](https://www.sfu.ca/~ssurjano/ackley2.png)
![Ackley's Function Image](https://static.wixstatic.com/media/40af5e_54cd6ec4d784436fa5e5c890cfb16c41~mv2.png/v1/fill/w_465,h_410,al_c,lg_1,q_85,enc_auto/40af5e_54cd6ec4d784436fa5e5c890cfb16c41~mv2.png)
#### Sphere Function !!
The Sphere function is deceptively simple, involving the sum of squares of its variables. It tests an algorithm's basic ability to hone in on a global minimum in a convex space without local minima distractions.

![SphereFunction](https://www.sfu.ca/~ssurjano/spheref2.png)
![Sphere Function Image](https://www.sfu.ca/~ssurjano/spheref.png)
#### Rastrigin's Function !!
Rastrigin's Function is a highly oscillatory function that provides a challenging landscape for optimization algorithms due to its large number of local minima. It's great for testing an algorithm's ability to find a global optimum amidst noise and local optima.
![Rastrigin's Function](https://www.sfu.ca/~ssurjano/rastr2.png)
![Rastrigin's Function Image](https://upload.wikimedia.org/wikipedia/commons/8/8b/Rastrigin_function.png)

#### Rosenbrock's Function !!
Also known as the Valley or Banana function, the Rosenbrock function is a non-convex function used to test the performance of optimization algorithms. It has a narrow, curved valley containing the global minimum, testing the algorithms' precision and convergence to the minimum.
![Rosenbrock's  Function](https://www.sfu.ca/~ssurjano/rosensc.png)
![Rosenbrock's  Function Image](https://www.sfu.ca/~ssurjano/rosen.png)

These functions were chosen for their ability to test the algorithms' exploration and exploitation capabilities, convergence speed, and robustness to local optima.

### Evaluation Metrics

Algorithms were assessed based on:

- **Accuracy**: The ability to locate the global optimum of benchmark functions.
- **Convergence Speed**: The number of iterations required to converge to a solution.
- **Robustness**: Consistency of performance across different functions.
