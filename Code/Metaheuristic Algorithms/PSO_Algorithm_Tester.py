# algorithm_performance_tester.py
from Particle_Swarm_Optimization import PSO
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json

# Parameters for PSO
num_particles = 30
iterations = 100
c1 = 1.49
c2 = 1.49
w = 0.729

# Define bounds for each function, assuming a 30-dimensional search space
bounds = [(-100, 100)] * 30  # Adjust these bounds as needed for each benchmark function

# Function to run PSO and return metrics
def run_pso_on_function(function, bounds):
    pso = PSO(function, bounds, num_particles, iterations, c1, c2, w)
    best_position, best_fitness, fitness_history = pso.run()
    average_fitness = np.mean(fitness_history)
    std_dev_fitness = np.std(fitness_history)
    # Assuming convergence when the change in fitness is small enough
    convergence_generation = next((i for i, fitness in enumerate(fitness_history) if fitness - best_fitness < 0.0001), iterations)
    return best_position, best_fitness, average_fitness, std_dev_fitness, convergence_generation, fitness_history

# Dictionary to store benchmark functions
benchmark_functions = {
    'sphere': sphere,
    'ackley': ackley,
    'rastrigin': rastrigin,
    'rosenbrock': rosenbrock
}

# Dictionary to store results
results = {}

# Running PSO on each benchmark function
for name, function in benchmark_functions.items():
    print(f"Running PSO on {name} function...")
    best_position, best_fitness, avg_fitness, std_dev, conv_gen, fitness_hist = run_pso_on_function(function, bounds)
    results[name] = {
        'Best Position': best_position.tolist(),  # Converting numpy array to list for JSON serialization
        'Best Fitness': best_fitness,
        'Average Fitness': avg_fitness,
        'Standard Deviation': std_dev,
        'Convergence Generation': conv_gen,
        'Fitness History': fitness_hist
    }

    # Plotting the fitness history for the function
    plt.figure()
    plt.plot(fitness_hist)
    plt.title(f'{name} Function Fitness Over Generations (PSO)')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.tight_layout()
    plt.savefig(f'{name}_pso_fitness_over_generations.png')
    plt.show()

# Output the results
for name, result in results.items():
    print(f"Results for {name}:")
    print(f"Best Position: {result['Best Position']}")
    print(f"Best Fitness: {result['Best Fitness']}")
    print(f"Average Fitness: {result['Average Fitness']}")
    print(f"Standard Deviation: {result['Standard Deviation']}")
    print(f"Convergence Generation: {result['Convergence Generation']}\n")

# Optionally: Save the results to a JSON file for later analysis
with open('pso_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4)
