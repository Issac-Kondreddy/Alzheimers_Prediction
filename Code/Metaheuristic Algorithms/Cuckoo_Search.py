import numpy as np
from scipy.special import gamma
class CuckooSearch:
    def __init__(self, objective_func, population_size, num_dimensions, num_nests, pa, alpha, generations):
        self.objective_func = objective_func
        self.population_size = population_size
        self.num_dimensions = num_dimensions
        self.num_nests = num_nests
        self.pa = pa
        self.alpha = alpha
        self.generations = generations
        self.nests = None
        self.fitness = None
        self.best_fitness_per_generation = []

    def initialize_nests(self):
        self.nests = np.random.rand(self.num_nests, self.num_dimensions)  # Initialize nests randomly
        self.fitness = np.zeros(self.num_nests)

    def evaluate_nests(self):
        for i in range(self.num_nests):
            self.fitness[i] = self.objective_func(self.nests[i])

    def levy_flight(self):
        beta = 3/2
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.num_dimensions) * sigma
        v = np.random.randn(self.num_dimensions)
        step = u / abs(v) ** (1 / beta)
        return self.alpha * step

    def search(self):
        self.initialize_nests()
        self.evaluate_nests()

        best_nest_idx = np.argmin(self.fitness)
        best_fitness_ever = self.fitness[best_nest_idx]

        for generation in range(self.generations):
            # Generate new solutions (eggs) by performing Levy flights
            for i in range(self.num_nests):
                step = self.levy_flight()
                new_nest = self.nests[i] + step * self.pa * (np.random.rand() - 0.5)  # Random walk
                new_nest = np.clip(new_nest, 0, 1)  # Ensure within bounds
                new_fitness = self.objective_func(new_nest)

                # Replace the old nest with the new one if the new one is better
                if new_fitness < self.fitness[i]:
                    self.nests[i] = new_nest
                    self.fitness[i] = new_fitness

            # Replace a portion of nests with new solutions (cuckoos)
            sorted_indices = np.argsort(self.fitness)
            num_replaced = int(self.pa * self.num_nests)
            cuckoos = np.random.rand(num_replaced, self.num_dimensions)
            for i in range(num_replaced):
                cuckoo_index = np.random.randint(0, num_replaced)
                replacement_index = sorted_indices[i]
                self.nests[replacement_index] = cuckoos[cuckoo_index]
                self.fitness[replacement_index] = self.objective_func(cuckoos[cuckoo_index])

            # Update best fitness
            best_nest_idx = np.argmin(self.fitness)
            best_fitness = self.fitness[best_nest_idx]
            if best_fitness < best_fitness_ever:
                best_fitness_ever = best_fitness
                self.best_fitness_per_generation.append(best_fitness)

        best_solution = self.nests[best_nest_idx]
        best_solution_fitness = self.fitness[best_nest_idx]

        return {
            'Best Solution': best_solution,
            'Best Fitness': best_solution_fitness,
            'Best Fitness Per Generation': self.best_fitness_per_generation,
            'Parameters': {
                'Population Size': self.population_size,
                'Number of Dimensions': self.num_dimensions,
                'Number of Nests': self.num_nests,
                'Pa': self.pa,
                'Alpha': self.alpha,
                'Generations': self.generations
            }
        }
