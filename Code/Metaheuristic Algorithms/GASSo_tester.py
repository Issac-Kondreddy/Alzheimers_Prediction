from GASSO import GASSO
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json

# GASSO Parameters
population_size = 50
iterations = 100
bounds = [(-100, 100)] * 30  # For a 30-dimensional search space


# Helper function to run GASSO on a given benchmark function
def run_gasso_on_function(function, bounds):
    gasso = GASSO(function, np.array(bounds), population_size, iterations)
    best_position, best_fitness, fitness_history = gasso.optimize()
    average_fitness, std_deviation, convergence_generation = gasso.calculate_metrics()

    # Ensure fitness history is a list of floats
    fitness_history = [float(f) for f in fitness_history]

    return best_fitness, average_fitness, std_deviation, convergence_generation, fitness_history


# Running GASSO on each benchmark function and collecting results
results = {}
for name, function in {'sphere': sphere, 'ackley': ackley, 'rastrigin': rastrigin, 'rosenbrock': rosenbrock}.items():
    print(f"Running GASSO on {name} function...")
    best_fitness, avg_fitness, std_dev, conv_gen, fitness_hist = run_gasso_on_function(function, bounds)
    results[name] = {
        'Best Fitness': best_fitness,
        'Average Fitness': avg_fitness,
        'Standard Deviation': std_dev,
        'Convergence Generation': conv_gen,
        'Fitness History': fitness_hist  # Already a list of floats
    }

    for func_name, metrics in results.items():
        for metric, value in metrics.items():
            if isinstance(value, np.ndarray):
                # Convert NumPy arrays to lists
                results[func_name][metric] = value.tolist()
            elif isinstance(value, (np.float64, np.int64)):
                # Convert NumPy floats and ints to Python native types
                results[func_name][metric] = float(value) if metric != 'Convergence Generation' else int(value)

    # Plot and save fitness history
    plt.figure()
    plt.plot(fitness_hist)
    plt.title(f'{name} Function Fitness Over Generations (GASSO)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name}_gasso_fitness_over_generations.png')
    plt.show()

# Serialize and save the results to JSON
with open('../optimization_curves/gasso_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4)


print("GASSO performance results saved to 'gasso_performance_results.json'.")
