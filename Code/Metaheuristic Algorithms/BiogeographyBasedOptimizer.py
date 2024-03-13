import numpy as np

class BiogeographyBasedOptimizer:
    def __init__(self, objective_func, population_size, dimension, migration_rate, mutation_rate, generations):
        self.objective_func = objective_func
        self.population_size = population_size
        self.dimension = dimension
        self.migration_rate = migration_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = None
        self.fitness = None
        self.best_fitness_per_generation = []  # Track best fitness per generation
        self.average_fitness_history = []  # Track average fitness per generation
        self.std_deviation_history = []  # Track standard deviation per generation
        self.convergence_generation = None

    def initialize_population(self):
        self.population = np.random.uniform(-5, 5, (self.population_size, self.dimension))
        self.fitness = np.zeros(self.population_size)

    def evaluate_population(self):
        for i in range(self.population_size):
            self.fitness[i] = self.objective_func(self.population[i])

    def migrate_population(self):
        migration_size = int(self.migration_rate * self.population_size)
        immigration_size = self.population_size - migration_size
        immigrant_indices = np.random.choice(self.population_size, immigration_size, replace=False)
        immigrant_population = np.random.uniform(-5, 5, (immigration_size, self.dimension))
        self.population[immigrant_indices] = immigrant_population

    def mutate_population(self):
        mutation_indices = np.random.rand(self.population_size, self.dimension) < self.mutation_rate
        self.population[mutation_indices] = np.random.uniform(-5, 5, np.sum(mutation_indices))

    def run(self):
        self.initialize_population()
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

            self.best_fitness_per_generation.append(best_fitness)
            self.average_fitness_history.append(average_fitness)
            self.std_deviation_history.append(std_deviation)

            self.migrate_population()
            self.mutate_population()

            if no_improvement_streak >= 10:  # Define convergence criteria here
                self.convergence_generation = generation + 1
                break

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
                'Dimension': self.dimension,
                'Migration Rate': self.migration_rate,
                'Mutation Rate': self.mutation_rate,
                'Generations': self.generations
            }
        }
