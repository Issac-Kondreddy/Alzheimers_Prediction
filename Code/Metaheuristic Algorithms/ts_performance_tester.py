from Tabu_search import TabuSearch  # Make sure to have this class defined as discussed
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock  # Assuming same benchmark functions
import numpy as np
import matplotlib.pyplot as plt
import json

# Tabu Search Parameters
num_iterations = 100
tabu_tenure = 15  # How long moves are forbidden
num_neighbors = 50  # How many neighbors to generate per iteration

# Bounds for the functions, assuming a 30-dimensional search space
bounds = [(-100, 100)] * 30

# Function to run TS and return metrics
def run_ts_on_function(function, bounds):
    ts = TabuSearch(function, bounds, num_iterations, tabu_tenure, num_neighbors)
    best_position, best_cost, fitness_history = ts.optimize()

    # Clean up fitness history - replace non-finite values with a large number to indicate poor fitness
    cleaned_fitness_history = [np.inf if not np.isfinite(cost) else cost for cost in fitness_history]

    average_cost = np.mean([cost for cost in cleaned_fitness_history if np.isfinite(cost)])
    std_dev_cost = np.std([cost for cost in cleaned_fitness_history if np.isfinite(cost)])
    convergence_generation = next(
        (i for i, cost in enumerate(cleaned_fitness_history) if np.isfinite(cost) and cost - best_cost < 0.0001),
        num_iterations)

    return best_position, best_cost, average_cost, std_dev_cost, convergence_generation, cleaned_fitness_history


# Running TS on each benchmark function
results = {}
for name, function in {'sphere': sphere, 'ackley': ackley, 'rastrigin': rastrigin, 'rosenbrock': rosenbrock}.items():
    print(f"Running TS on {name} function...")
    best_position, best_cost, avg_cost, std_dev, conv_gen, fitness_hist = run_ts_on_function(function, bounds)
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
    plt.title(f'{name} Function Fitness Over Generations (TS)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.tight_layout()
    plt.savefig(f'{name}_ts_fitness_over_generations.png')
    plt.show()

# Output or save results as needed
# Before dumping the results to JSON, ensure all NumPy arrays are converted to lists
def convert_to_list(item):
    if isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, list):
        return [convert_to_list(subitem) for subitem in item]
    elif isinstance(item, dict):
        return {key: convert_to_list(value) for key, value in item.items()}
    else:
        return item

# Apply conversion to your results before dumping to JSON
results_serializable = convert_to_list(results)

# Now, it should be safe to dump the results to JSON
with open('../optimization_curves/ts_performance_results.json', 'w') as fp:
    json.dump(results_serializable, fp, indent=4)

print("TS performance results saved to 'ts_performance_results.json'.")



print("TS performance results saved to 'ts_performance_results.json'.")
