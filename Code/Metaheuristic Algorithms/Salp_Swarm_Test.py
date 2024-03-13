from Salp_Swarm_Algorithm import SalpSwarmAlgorithm  # Assuming you have saved SalpSwarmAlgorithm class in SalpSwarmAlgorithm.py
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json

# Parameters for the Salp Swarm Algorithm
population_size = 100
dimension = 30  # Adjust based on your benchmark function requirements
step_size = 0.1
influence_coefficient = 0.1
generations = 100

# Benchmark functions to test
benchmark_functions = {
    "Sphere": sphere,
    "Ackley": ackley,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}


def run_salp_swarm_on_function(function_name, function):
    print(f"Running Salp Swarm algorithm on {function_name} function...")
    salp_swarm_algo = SalpSwarmAlgorithm(
        objective_func=function,
        population_size=population_size,
        dimension=dimension,
        step_size=step_size,
        influence_coefficient=influence_coefficient,
        generations=generations
    )

    # Run the Salp Swarm algorithm
    results = salp_swarm_algo.run()

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
        'Average Fitness': np.mean(results['Average Fitness History']),
        'Standard Deviation': np.std(results['Standard Deviation History']),
        'Convergence Generation': results['Convergence Generation'],
        'Parameters': results['Parameters'],
        'Fitness History': [float(f) for f in results['Best Fitness Per Generation']]  # Convert to a list of floats
    }


# Dictionary to store the results
results = {}

# Running the Salp Swarm algorithm on each benchmark function and storing the results
for name, function in benchmark_functions.items():
    try:
        results[name] = run_salp_swarm_on_function(name, function)
    except Exception as e:
        print(f"An error occurred while running the Salp Swarm algorithm on the {name} function: {e}")

# Optional: Save the results to a JSON file for later analysis
with open('salp_swarm_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4)
