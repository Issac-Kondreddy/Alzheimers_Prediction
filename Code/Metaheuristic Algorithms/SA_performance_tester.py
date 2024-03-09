from Simulated_Annealing import simulated_annealing
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Define dynamic parameters for each benchmark function
sa_params = {
    'sphere': {'initial_temp': 10000, 'final_temp': 1, 'alpha': 0.99, 'max_iterations': 1000},
    'ackley': {'initial_temp': 5000, 'final_temp': 1, 'alpha': 0.98, 'max_iterations': 1500},
    'rastrigin': {'initial_temp': 8000, 'final_temp': 1, 'alpha': 0.95, 'max_iterations': 1200},
    'rosenbrock': {'initial_temp': 10000, 'final_temp': 1, 'alpha': 0.97, 'max_iterations': 1500},
}

# Bounds for the functions, assuming a 30-dimensional search space
bounds = [(-100, 100)] * 30

def run_sa_on_function(function, bounds, initial_temp, final_temp, alpha, max_iterations):
    best_position, best_cost, fitness_history = simulated_annealing(
        function, bounds, initial_temp, final_temp, alpha, max_iterations
    )
    average_cost = np.mean(fitness_history)
    std_dev_cost = np.std(fitness_history)
    convergence_generation = next(
        (i for i, cost in enumerate(fitness_history) if cost - best_cost < 0.0001),
        max_iterations
    )
    return {
        'best_position': best_position.tolist(),
        'best_cost': float(best_cost),
        'average_cost': float(average_cost),
        'std_dev_cost': float(std_dev_cost),
        'convergence_generation': convergence_generation,
        'fitness_history': [float(cost) for cost in fitness_history]
    }

# Prepare to store and visualize results
results = {}
output_dir = '../optimization_curves/'
os.makedirs(output_dir, exist_ok=True)

# Running Simulated Annealing on each benchmark function with customized parameters
for name, function in {'sphere': sphere, 'ackley': ackley, 'rastrigin': rastrigin, 'rosenbrock': rosenbrock}.items():
    print(f"Running Simulated Annealing on {name} function...")
    params = sa_params[name]
    result = run_sa_on_function(function, bounds, **params)
    results[name] = result

    # Plot and save fitness history
    plt.figure()
    plt.plot(result['fitness_history'])
    plt.title(f'{name} Function Fitness Over Iterations (Simulated Annealing)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}_sa_fitness_over_iterations.png'))
    plt.close()

# Save results to a JSON file
with open(os.path.join(output_dir, 'sa_performance_results.json'), 'w') as fp:
    json.dump(results, fp, indent=4)

print("Simulated Annealing performance results saved.")
