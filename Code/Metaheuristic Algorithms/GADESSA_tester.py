from GADESSA import DEGASSA  # Ensure DEGASSA class is correctly imported
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock  # Import your benchmark functions
import matplotlib.pyplot as plt
import json

def run_degassa_on_function(function, bounds, population_size=50, iterations=100):
    degassa = DEGASSA(function, bounds, population_size, iterations)
    best_solution, best_fitness = degassa.optimize()
    avg_fitness, std_dev, conv_gen = degassa.calculate_metrics()
    return best_solution, best_fitness, avg_fitness, std_dev, conv_gen, degassa.fitness_history

# Define your search space bounds
bounds = [(-100, 100)] * 30  # Example: 30-dimensional problem

results = {}
for name, function in {'sphere': sphere, 'ackley': ackley, 'rastrigin': rastrigin, 'rosenbrock': rosenbrock}.items():
    print(f"Running DEGASSA on {name} function...")
    best_solution, best_fitness, avg_fitness, std_dev, conv_gen, fitness_history = run_degassa_on_function(function, bounds)
    results[name] = {
        'Best Solution': best_solution.tolist(),
        'Best Fitness': best_fitness,
        'Average Fitness': avg_fitness,
        'Standard Deviation': std_dev,
        'Convergence Generation': conv_gen if conv_gen is not None else "Not Converged",
        'Fitness History': fitness_history
    }

    # Plot the fitness history
    plt.figure()
    plt.plot(fitness_history, label='Fitness over Generations')
    plt.title(f'Fitness History for {name}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.savefig(f'{name}_fitness_history.png')
    plt.close()

# Optionally, save the results to a JSON file for later analysis
with open('DEGASSA_results.json', 'w') as file:
    json.dump(results, file, indent=4)

print("DEGASSA testing complete. Results saved.")
