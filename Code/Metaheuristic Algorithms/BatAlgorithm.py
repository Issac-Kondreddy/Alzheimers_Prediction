import numpy as np

class BATAlgorithm:
    def __init__(self, objective_func, population_size, dimension, pulse_rate, loudness, generations):
        self.objective_func = objective_func
        self.population_size = population_size
        self.dimension = dimension
        self.pulse_rate = pulse_rate
        self.loudness = loudness
        self.generations = generations
        self.population = None
        self.fitness = None
        self.best_fitness_per_generation = []  # Track best fitness per generation

    def initialize_population(self):
        self.population = np.random.uniform(-5, 5, (self.population_size, self.dimension))
        self.fitness = np.zeros(self.population_size)

    def evaluate_population(self):
        for i in range(self.population_size):
            self.fitness[i] = self.objective_func(self.population[i])

    def update_loudness(self, generation):
        return self.loudness * (1 - np.exp(-generation))

    def update_pulse_rate(self, generation):
        return self.pulse_rate * (1 - np.exp(-generation))

    def move_bats(self, generation):
        updated_population = np.copy(self.population)
        for i in range(self.population_size):
            loudness_i = self.update_loudness(generation)
            pulse_rate_i = self.update_pulse_rate(generation)

            for j in range(self.dimension):
                updated_population[i, j] += np.random.uniform(-1, 1) * pulse_rate_i

                if np.random.rand() > self.pulse_rate:
                    updated_population[i, j] = np.random.uniform(-5, 5)

            if np.random.rand() < loudness_i and self.objective_func(updated_population[i]) < self.objective_func(self.population[i]):
                self.population[i] = updated_population[i]

    def run(self):
        self.initialize_population()
        best_fitness_ever = float('inf')
        no_improvement_streak = 0  # Tracks the number of generations without improvement

        for generation in range(self.generations):
            self.evaluate_population()
            best_fitness = np.min(self.fitness)
            if best_fitness < best_fitness_ever:
                best_fitness_ever = best_fitness
                no_improvement_streak = 0  # Reset if there's improvement
            else:
                no_improvement_streak += 1  # Increment if there's no improvement

            self.best_fitness_per_generation.append(best_fitness)

            self.move_bats(generation)

        best_overall_idx = np.argmin(self.fitness)
        best_solution = self.population[best_overall_idx]
        best_solution_fitness = self.fitness[best_overall_idx]

        return {
            'Best Solution': best_solution,
            'Best Fitness': best_solution_fitness,
            'Best Fitness Per Generation': self.best_fitness_per_generation,
            'Parameters': {
                'Population Size': self.population_size,
                'Dimension': self.dimension,
                'Pulse Rate': self.pulse_rate,
                'Loudness': self.loudness,
                'Generations': self.generations
            }
        }
