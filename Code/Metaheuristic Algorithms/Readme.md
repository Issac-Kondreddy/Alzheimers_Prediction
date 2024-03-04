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
- Equation: \(f(\mathbf{x}) = -20\exp\left(-0.2\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}\right) - \exp\left(\frac{1}{n}\sum_{i=1}^{n}\cos(2\pi x_i)\right) + 20 + e\)

#### Sphere Function !!
The Sphere function is deceptively simple, involving the sum of squares of its variables. It tests an algorithm's basic ability to hone in on a global minimum in a convex space without local minima distractions.
- Equation: \(f(\mathbf{x}) = \sum_{i=1}^{n} x_i^2\)

#### Rastrigin's Function !!
Rastrigin's Function is a highly oscillatory function that provides a challenging landscape for optimization algorithms due to its large number of local minima. It's great for testing an algorithm's ability to find a global optimum amidst noise and local optima.
- Equation: \(f(\mathbf{x}) = 10n + \sum_{i=1}^{n}\left[x_i^2 - 10\cos(2\pi x_i)\right]\)

#### Rosenbrock's Function !!
Also known as the Valley or Banana function, the Rosenbrock function is a non-convex function used to test the performance of optimization algorithms. It has a narrow, curved valley containing the global minimum, testing the algorithms' precision and convergence to the minimum.
- Equation: \(f(\mathbf{x}) = \sum_{i=1}^{n-1} \left[100(x_{i+1} - x_i^2)^2 + (1-x_i)^2\right]\)


These functions were chosen for their ability to test the algorithms' exploration and exploitation capabilities, convergence speed, and robustness to local optima.

### Evaluation Metrics

Algorithms were assessed based on:

- **Accuracy**: The ability to locate the global optimum of benchmark functions.
- **Convergence Speed**: The number of iterations required to converge to a solution.
- **Robustness**: Consistency of performance across different functions.
