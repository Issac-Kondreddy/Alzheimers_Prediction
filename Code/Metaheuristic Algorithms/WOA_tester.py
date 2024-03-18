from WhaleOptimizationAlgorithm import WhaleOptimizationAlgorithm
from Benchmark_functions import sphere, ackley, rastrigin, rosenbrock
import numpy as np
import json
import matplotlib.pyplot as plt

# Define NumpyEncoder for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

benchmark_functions = {
    "Sphere": sphere,
    "Ackley": ackley,
    "Rastrigin": rastrigin,
    "Rosenbrock": rosenbrock
}

lb, ub, dim = -100, 100, 30
results = {}

for name, function in benchmark_functions.items():
    woa = WhaleOptimizationAlgorithm(objective_func=function, lb=lb, ub=ub, dim=dim)
    result = woa.optimize()
    results[name] = result

    plt.figure()
    plt.plot(result['Fitness History'])
    plt.title(f'{name} Function Cost Over Iterations (WOA)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.savefig(f'{name}_WOA_cost_over_iterations.png')
    plt.close()

with open('woa_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4, cls=NumpyEncoder)

print("Whale Optimization Algorithm performance results saved.")
