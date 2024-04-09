from GADE import GDE
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json


def run_gde_on_function(function, bounds, population_size=50, iterations=100):
    gde = GDE(function, bounds, population_size, iterations)
    gde.optimize()  # Perform the optimization

    # Ensure metrics are calculated correctly
    avg_fitness, std_dev, conv_gen = gde.calculate_metrics()
    best_solution, best_fitness = gde.evaluate()

    # Check for potential issues with metric calculation
    if avg_fitness is None or std_dev is None or conv_gen is None:
        print("Error in calculating metrics. Check fitness_history for issues.")

    return best_solution, best_fitness, avg_fitness, std_dev, conv_gen, gde.fitness_history


results = {}
for name, function in {'sphere': sphere, 'ackley': ackley, 'rastrigin': rastrigin, 'rosenbrock': rosenbrock}.items():
    print(f"Running GDE on {name} function...")
    best_solution, best_fitness, avg_fitness, std_dev, conv_gen, fitness_history = run_gde_on_function(function, [
        (-100, 100)] * 30)
    results[name] = {
        'Best Solution': best_solution.tolist(),  # Ensure solution is list
        'Best Fitness': float(best_fitness),  # Ensure fitness is a float
        'Average Fitness': avg_fitness,
        'Standard Deviation': std_dev,
        'Convergence Generation': conv_gen,
        'Fitness History': fitness_history  # Directly store the history
    }

    # Plotting fitness history for visualization
    plt.figure()
    plt.plot(fitness_history, label='Fitness Over Generations')
    plt.title(f'{name} Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(f'{name}_GDE_fitness.png')
    plt.close()

# Save results to JSON
with open('../optimization_curves/GDE_performance_results.json', 'w') as file:
    json.dump(results, file, indent=4)

print("GDE performance results saved.")
