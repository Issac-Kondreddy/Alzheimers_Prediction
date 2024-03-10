from Differential_Evolution import DifferentialEvolution  # Assuming you named the file as Differential_Evolution.py
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import matplotlib.pyplot as plt
import numpy as np
import json

# Parameters for Differential Evolution
population_size = 100
chromosome_length = 30
F = 0.8  # Differential weight
CR = 0.9  # Crossover probability
generations = 100

benchmark_functions = {
    "Sphere": sphere,
    "Ackley": ackley,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}

def run_de_on_function(function_name, function):
    print(f"Running DE on {function_name} function...")
    de = DifferentialEvolution(
        objective_func=function,
        population_size=population_size,
        chromosome_length=chromosome_length,
        F=F,
        CR=CR,
        generations=generations
    )

    results = de.run()

    print(f"Best Fitness for {function_name}: {results['Best Fitness']}\n")

    plt.figure()
    plt.plot(results['Best Fitness Per Generation'])
    plt.title(f'{function_name} Function Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.tight_layout()
    plt.savefig(f'{function_name}_DE_fitness_over_generations.png')
    plt.show()

    return {
        'Best Solution': results['Best Solution'].tolist(),
        'Best Fitness': results['Best Fitness'],
        'Average Fitness': np.mean(results['Average Fitness History']),
        'Standard Deviation': np.std(results['Standard Deviation History']),
        'Parameters': results['Parameters'],
        'Fitness History': [float(f) for f in results['Best Fitness Per Generation']]
    }

results = {}

for name, function in benchmark_functions.items():
    try:
        results[name] = run_de_on_function(name, function)
    except Exception as e:
        print(f"Error running DE on {name} function: {e}")

with open('de_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4)
