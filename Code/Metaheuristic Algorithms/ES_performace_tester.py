from Evolutionary_Strategies import EvolutionaryStrategies
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import json
import matplotlib.pyplot as plt

benchmark_functions = {
    "Sphere": sphere,
    "Ackley": ackley,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

bounds = np.array([[-100, 100]] * 30)  # Example for 30-dimensional optimization problems

results = {}
for name, function in benchmark_functions.items():
    es = EvolutionaryStrategies(objective_func=function, bounds=bounds, population_size=50, sigma=0.1, learning_rate=0.001, max_iters=1000)
    result = es.run()
    results[name] = {k: np.array(v).tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
    plt.figure()
    plt.plot(result['Fitness History'])
    plt.title(f'{name} Function Fitness Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.savefig(f'{name}_fitness_over_generations.png')
    plt.close()

with open('es_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4, cls=NumpyEncoder)

print("Evolutionary Strategies performance results saved.")
