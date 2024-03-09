# algorithm_performance_tester.py
from Particle_Swarm_Optimization import PSO
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import matplotlib.pyplot as plt
import json

# Parameters for PSO
num_particles = 300
iterations = 1000
c1 = 1.56
c2 = 1.56
w = 0.66

# Define bounds for each function, assuming a 30-dimensional search space
bounds = [(-1000, 1000)] * 45  # Adjust these bounds as needed for each benchmark function
function_params = {
    'sphere': {'num_particles': 100, 'iterations': 500, 'c1': 2.05, 'c2': 2.05, 'w': 0.729},
    'ackley': {'num_particles': 150, 'iterations': 700, 'c1': 1.8, 'c2': 1.8, 'w': 0.6},
    'rastrigin': {'num_particles': 200, 'iterations': 800, 'c1': 1.5, 'c2': 1.5, 'w': 0.7},
    'rosenbrock': {'num_particles': 250, 'iterations': 1000, 'c1': 2.0, 'c2': 2.0, 'w': 0.5}
}


# Function to run PSO and return metrics
def run_pso_on_function(function, bounds, params):
    pso = PSO(function, bounds, params['num_particles'], params['iterations'], params['c1'], params['c2'], params['w'])
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
# Running PSO on each benchmark function
for name, function in benchmark_functions.items():
    print(f"Running PSO on {name} function...")
    params = function_params[name]  # Retrieve the specific parameters for the current function
    best_position, best_fitness, avg_fitness, std_dev, conv_gen, fitness_hist = run_pso_on_function(function, bounds, params)  # Pass the parameters
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
