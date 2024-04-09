from DESSA import DESSA
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json

# Parameters for the DESSA Algorithm
population_size = 50
iterations = 100
bounds = [(-100, 100)] * 30  # Adjust based on your benchmark function requirements

# Running DESSA on each benchmark function and storing the results
results = {}
for name, function in {'sphere': sphere, 'ackley': ackley, 'rastrigin': rastrigin, 'rosenbrock': rosenbrock}.items():
    print(f"Running DESSA on {name} function...")
    dessa = DESSA(function, bounds, population_size, iterations)
    best_solution, best_fitness, fitness_history = dessa.optimize()
    avg_fitness, std_dev, conv_gen = dessa.calculate_metrics()

    results[name] = {
        'Best Fitness': best_fitness,
        'Average Fitness': avg_fitness,
        'Standard Deviation': std_dev,
        'Convergence Generation': conv_gen if conv_gen != -1 else "Not Converged",
        'Fitness History': fitness_history
    }

    plt.figure()
    plt.plot(fitness_history)
    plt.title(f'{name} Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig(f'{name}_DESSA_fitness.png')
    plt.close()

# Convert results to JSON serializable format
for function_name in results:
    for key in results[function_name]:
        if isinstance(results[function_name][key], np.ndarray):
            # Convert the ndarray to a list
            results[function_name][key] = results[function_name][key].tolist()
        elif isinstance(results[function_name][key], list):
            # Ensure all elements in lists are also converted, in case of nested structures
            results[function_name][key] = [item.tolist() if isinstance(item, np.ndarray) else item for item in results[function_name][key]]

# Now, save the converted results to a JSON file
with open('../optimization_curves/DESSA_performance_results.json', 'w') as file:
    json.dump(results, file, indent=4)

print("DESSA performance results saved.")


