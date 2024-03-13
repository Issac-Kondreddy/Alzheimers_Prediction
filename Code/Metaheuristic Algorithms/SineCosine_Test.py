from SineCosine_Algorithm import SineCosineAlgorithm  # Assuming you have saved SineCosineAlgorithm class in SineCosineAlgorithm.py
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json

# Parameters for the Sine Cosine Algorithm
population_size = 100
dimension = 30  # Adjust based on your benchmark function requirements
generations = 100
a = 3  # Parameter 'a' for sine
r = 1  # Parameter 'r' for cosine

# Benchmark functions to test
benchmark_functions = {
    "Sphere": sphere,
    "Ackley": ackley,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}

def run_sine_cosine_on_function(function_name, function):
    print(f"Running Sine Cosine algorithm on {function_name} function...")
    sine_cosine_algo = SineCosineAlgorithm(
        objective_func=function,
        population_size=population_size,
        dimension=dimension,
        generations=generations,
        a=a,
        r=r
    )

    # Run the Sine Cosine algorithm
    results = sine_cosine_algo.run()

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

# Running the Sine Cosine algorithm on each benchmark function and storing the results
for name, function in benchmark_functions.items():
    try:
        results[name] = run_sine_cosine_on_function(name, function)
    except Exception as e:
        print(f"An error occurred while running the Sine Cosine algorithm on the {name} function: {e}")

# Optional: Save the results to a JSON file for later analysis
with open('sine_cosine_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4)
