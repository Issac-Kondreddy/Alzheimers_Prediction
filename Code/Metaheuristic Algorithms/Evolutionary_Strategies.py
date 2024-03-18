import numpy as np

class EvolutionaryStrategies:
    def __init__(self, objective_func, bounds, population_size=50, sigma=0.1, learning_rate=0.001, max_iters=100):
        self.objective_func = objective_func
        self.bounds = bounds
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.max_iters = max_iters

    def mutate(self, parent):
        child = parent + np.random.normal(0, self.sigma, size=parent.shape)
        child = np.clip(child, a_min=self.bounds[:, 0], a_max=self.bounds[:, 1])
        return child

    def run(self):
        dimension = len(self.bounds)
        best_solution = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=dimension)
        best_fitness = self.objective_func(best_solution)
        fitness_history = [best_fitness]

        for generation in range(self.max_iters):
            children = np.array([self.mutate(best_solution) for _ in range(self.population_size)])
            fitnesses = np.array([self.objective_func(child) for child in children])
            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < best_fitness:
                best_fitness = fitnesses[best_idx]
                best_solution = children[best_idx]
            fitness_history.append(best_fitness)

        average_fitness = np.mean(fitness_history)
        std_dev_fitness = np.std(fitness_history)
        convergence_generation = np.argmin(fitness_history)
        parameters = {
            'Population Size': self.population_size,
            'Sigma': self.sigma,
            'Learning Rate': self.learning_rate,
            'Max Iters': self.max_iters
        }

        return {
            'Best Solution': best_solution,
            'Best Fitness': best_fitness,
            'Average Fitness': average_fitness,
            'Standard Deviation': std_dev_fitness,
            'Convergence Generation': convergence_generation,
            'Parameters': parameters,
            'Fitness History': fitness_history
        }
