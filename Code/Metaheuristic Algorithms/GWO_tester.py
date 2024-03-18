from GreyWolfOptimizer import GreyWolfOptimizer
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
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


lb, ub, dim = -100, 100, 30
results = {}

for name, function in benchmark_functions.items():
    gwo = GreyWolfOptimizer(objective_func=function, lb=lb, ub=ub, dim=dim)
    result = gwo.optimize()
    result['Best Position'] = result['Best Position'].tolist()
    result['Fitness History'] = [float(f) for f in result['Fitness History']]
    results[name] = result

    plt.figure()
    plt.plot(result['Fitness History'])
    plt.title(f'{name} Function Cost Over Iterations (GWO)')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.savefig(f'{name}_GWO_cost_over_iterations.png')
    plt.close()

with open('gwo_performance_results.json', 'w') as fp:
    json.dump(results, fp, indent=4, cls=NumpyEncoder)

print("GWO performance results saved.")