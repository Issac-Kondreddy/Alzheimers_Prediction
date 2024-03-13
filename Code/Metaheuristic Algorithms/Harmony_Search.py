import numpy as np

class HarmonySearch:
    def __init__(self, objective_func, population_size, dimension, harmony_memory_size, pitch_adjustment_rate, bandwidth, generations):
        self.objective_func = objective_func
        self.population_size = population_size
        self.dimension = dimension
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.bandwidth = bandwidth
        self.generations = generations
        self.population = None
        self.fitness = None
        self.best_fitness_per_generation = []  # Track best fitness per generation
        self.average_fitness_history = []  # Track average fitness per generation
        self.std_deviation_history = []  # Track standard deviation per generation
        self.convergence_generation = None  # Track convergence generation

    def initialize_population(self):
        self.population = np.random.uniform(-5, 5, (self.population_size, self.dimension))
        self.fitness = np.zeros(self.population_size)

    def evaluate_population(self):
        for i in range(self.population_size):
            self.fitness[i] = self.objective_func(self.population[i])

    def move_harmonies(self):
        updated_population = np.copy(self.population)
        for i in range(self.population_size):
            for j in range(self.dimension):
                if np.random.rand() < self.pitch_adjustment_rate:
                    index = np.random.randint(0, self.harmony_memory_size)
                    updated_population[i, j] = self.population[index, j] + np.random.uniform(-self.bandwidth, self.bandwidth)

        self.population = np.clip(updated_population, -5, 5)

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

            self.best_fitness_per_generation.append(best_fitness)
            self.average_fitness_history.append(average_fitness)
            self.std_deviation_history.append(std_deviation)
            self.move_harmonies()

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
                'Harmony Memory Size': self.harmony_memory_size,
                'Pitch Adjustment Rate': self.pitch_adjustment_rate,
                'Bandwidth': self.bandwidth,
                'Generations': self.generations
            }
        }
