from Cuckoo_Search import CuckooSearch  # Make sure it returns best_fitness_per_generation
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json

# Parameters for the Cuckoo Search algorithm
population_size = 100
num_dimensions = 30  # Adjust based on your benchmark function requirements
num_nests = 10
pa = 0.25
alpha = 0.01
generations = 100

# Benchmark functions to test
benchmark_functions = {
    "Sphere": sphere,
    "Ackley": ackley,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}


def run_cs_on_function(function_name, function):
    print(f"Running CS on {function_name} function...")
    cs = CuckooSearch(
        objective_func=function,
        population_size=population_size,
        num_dimensions=num_dimensions,
        num_nests=num_nests,
        pa=pa,
        alpha=alpha,
        generations=generations
    )

    # Run the CS algorithm
    results = cs.search()

    print(f"Best Fitness for {function_name}: {results['Best Fitness']}\n")

    # Plotting the fitness history for the function
    plt.figure()
    plt.plot(results['Best Fitness Per Generation'])
    plt.title(f'{function_name} Function Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.tight_layout()
    plt.savefig(f'{function_name}_fitness_over_generations.png')  # Save the plot as an image file
    plt.show()

    # No need to calculate the mean and std again, just pull them from the results
    return {
        'Best Solution': results['Best Solution'].tolist(),
        'Best Fitness': results['Best Fitness'],
        'Best Fitness Per Generation': results['Best Fitness Per Generation'],
        'Parameters': results['Parameters']
    }


# Dictionary to store the results
results = {}

# Running the CS algorithm on each benchmark function and storing the results
for name, function in benchmark_functions.items():
    try:
        results[name] = run_cs_on_function(name, function)
    except Exception as e:
        print(f"An error occurred while running the CS on the {name} function: {e}")

# Optional: Save the results to a JSON file for later analysis
with open('cs_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4)
