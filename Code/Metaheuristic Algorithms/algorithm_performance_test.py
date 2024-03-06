from Genetic_Algorithm import GeneticAlgorithm  # Make sure it returns best_fitness_per_generation
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json

# Parameters for the Genetic Algorithm
population_size = 100
chromosome_length = 30  # Adjust based on your benchmark function requirements
crossover_rate = 0.8
mutation_rate = 0.01
generations = 100

# Benchmark functions to test
benchmark_functions = {
    "Sphere": sphere,
    "Ackley": ackley,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}

# Function to run GA on a specific benchmark function
# ... (rest of your imports and setup)

def run_ga_on_function(function_name, function):
    print(f"Running GA on {function_name} function...")
    ga = GeneticAlgorithm(
        objective_func=function,
        population_size=population_size,
        chromosome_length=chromosome_length,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        generations=generations
    )
    # Run the GA and get all the additional information
    best_solution, best_fitness, fitness_history, avg_fitness_history, std_dev_history, conv_gen = ga.run()
    print(f"Best Fitness for {function_name}: {best_fitness}\n")
    
    # Plotting the fitness history for the function
    plt.figure()
    plt.plot(fitness_history)
    plt.title(f'{function_name} Function Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.tight_layout()
    plt.savefig(f'{function_name}_fitness_over_generations.png')  # Save the plot as an image file
    plt.show()
    
    # Return the results including the new metrics
    return {
        'Best Solution': best_solution.tolist(),
        'Best Fitness': best_fitness,
        'Average Fitness': np.mean(avg_fitness_history),
        'Standard Deviation': np.std(std_dev_history),
        'Convergence Generation': conv_gen,
        'Parameters': {
            'Population Size': population_size,
            'Chromosome Length': chromosome_length,
            'Crossover Rate': crossover_rate,
            'Mutation Rate': mutation_rate,
            'Generations': generations
        },
        'Fitness History': [float(f) for f in fitness_history]  # Convert to a list of floats
    }


# Dictionary to store the results
results = {}

# Running the GA on each benchmark function and storing the results
for name, function in benchmark_functions.items():
    try:
        results[name] = run_ga_on_function(name, function)
    except Exception as e:
        print(f"An error occurred while running the GA on the {name} function: {e}")

# Optional: Save the results to a JSON file for later analysis
with open('ga_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4)