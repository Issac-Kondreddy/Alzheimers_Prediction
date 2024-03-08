from Ant_Colony_Optimization import ACO
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json

# ACO Parameters
num_ants = 30
iterations = 100
pheromone_evaporation_coeff = 0.6
pheromone_constant = 1.0
alpha = 0.9  # Pheromone importance
beta = 1  # Heuristic importance

# Bounds for the functions, assuming a 30-dimensional search space
bounds = [(-100, 100)] * 30

# Function to run ACO and return metrics
def run_aco_on_function(function, bounds):
    aco = ACO(function, bounds, num_ants, iterations, pheromone_evaporation_coeff, pheromone_constant, alpha, beta)
    best_position, best_cost, fitness_history = aco.optimize()
    average_cost = np.mean(fitness_history)
    std_dev_cost = np.std(fitness_history)
    convergence_generation = next((i for i, cost in enumerate(fitness_history) if cost - best_cost < 0.0001), iterations)
    return best_position, best_cost, average_cost, std_dev_cost, convergence_generation, fitness_history

# Running ACO on each benchmark function
results = {}
for name, function in {'sphere': sphere, 'ackley': ackley, 'rastrigin': rastrigin, 'rosenbrock': rosenbrock}.items():
    print(f"Running ACO on {name} function...")
    best_position, best_cost, avg_cost, std_dev, conv_gen, fitness_hist = run_aco_on_function(function, bounds)
    results[name] = {
        'Best Position': best_position.tolist(),
        'Best Cost': best_cost,
        'Average Cost': avg_cost,
        'Standard Deviation': std_dev,
        'Convergence Generation': conv_gen,
        'Fitness History': fitness_hist
    }

    # Plot and save fitness history
    plt.figure()
    plt.plot(fitness_hist)
    plt.title(f'{name} Function Fitness Over Generations (ACO)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.tight_layout()
    plt.savefig(f'{name}_aco_fitness_over_generations.png')
    plt.show()

# Output or save results as needed
with open('../optimization_curves/aco_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4)

print("ACO performance results saved to 'aco_performance_results.json'.")