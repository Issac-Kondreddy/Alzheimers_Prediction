import numpy as np

class GeneticAlgorithm:
    def __init__(self, objective_func, population_size, chromosome_length, crossover_rate, mutation_rate, generations):
        self.objective_func = objective_func
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = None
        self.fitness = None
        self.best_fitness_per_generation = []  # Track best fitness per generation

    def initialize_population(self):
        self.population = np.random.rand(self.population_size, self.chromosome_length)
        self.fitness = np.zeros(self.population_size)

    def evaluate_population(self):
        for i in range(self.population_size):
            self.fitness[i] = self.objective_func(self.population[i])

    def select_parents(self):
        # Tournament selection
        parents = np.empty((self.population_size, self.chromosome_length))
        for i in range(self.population_size):
            random_dad = np.random.randint(0, self.population_size)
            random_mom = np.random.randint(0, self.population_size)
            dad_fitness = self.fitness[random_dad]
            mom_fitness = self.fitness[random_mom]
            if dad_fitness < mom_fitness:  # Assuming minimization
                parents[i, :] = self.population[random_dad, :]
            else:
                parents[i, :] = self.population[random_mom, :]
        return parents

    def crossover(self, parents):
        offspring = np.empty((self.population_size, self.chromosome_length))
        for k in range(0, self.population_size, 2):
            if np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.chromosome_length-1)
                offspring[k, :] = np.concatenate([parents[k, :crossover_point], parents[k+1, crossover_point:]])
                offspring[k+1, :] = np.concatenate([parents[k+1, :crossover_point], parents[k, crossover_point:]])
            else:
                offspring[k, :] = parents[k, :]
                offspring[k+1, :] = parents[k+1, :]
        return offspring

    def mutate(self, offspring):
        for idx in range(offspring.shape[0]):
            for gene in range(offspring.shape[1]):
                if np.random.rand() < self.mutation_rate:
                    random_value = np.random.uniform(-0.5, 0.5)
                    offspring[idx, gene] += random_value
        return offspring

    def run(self):
        self.initialize_population()
        self.best_fitness_per_generation = []  # Reset at the start of a new run
        self.average_fitness_history = []
        self.std_deviation_history = []
        self.convergence_generation = None
        best_fitness_ever = float('inf')
        no_improvement_streak = 0  # Tracks the number of generations without improvement

        for generation in range(self.generations):
            self.evaluate_population()
            best_fitness = np.min(self.fitness)
            average_fitness = np.mean(self.fitness)
            std_deviation = np.std(self.fitness)
            if best_fitness < best_fitness_ever:
                best_fitness_ever = best_fitness
                no_improvement_streak = 0  # Reset if there's improvement
            else:
                no_improvement_streak += 1  # Increment if there's no improvement

            # Update the convergence generation if the no improvement streak reaches a certain threshold
            if no_improvement_streak >= 10:  # For example, after 10 generations of no improvement
                if self.convergence_generation is None:
                    self.convergence_generation = generation - 10  # Subtracting 10 because that's when the improvement stopped

            self.best_fitness_per_generation.append(best_fitness)
            self.average_fitness_history.append(average_fitness)
            self.std_deviation_history.append(std_deviation)

            parents = self.select_parents()
            offspring_crossover = self.crossover(parents)
            offspring_mutation = self.mutate(offspring_crossover)
            self.population = offspring_mutation
            self.evaluate_population()

        best_overall_idx = np.argmin(self.fitness)
        best_solution = self.population[best_overall_idx]
        best_solution_fitness = self.fitness[best_overall_idx]

        return {
            'Best Solution': best_solution,
            'Best Fitness': best_solution_fitness,
            'Best Fitness Per Generation': self.best_fitness_per_generation,
            'Average Fitness History': self.average_fitness_history,
            'Standard Deviation History': self.std_deviation_history,
            'Convergence Generation': self.convergence_generation if self.convergence_generation else self.generations,
            'Parameters': {
                'Population Size': self.population_size,
                'Chromosome Length': self.chromosome_length,
                'Crossover Rate': self.crossover_rate,
                'Mutation Rate': self.mutation_rate,
                'Generations': self.generations
            }
        }


