import numpy as np

class GrasshopperAlgorithm:
    def __init__(self, objective_func, population_size, dimension, step_size, attraction_coefficient, generations):
        self.objective_func = objective_func
        self.population_size = population_size
        self.dimension = dimension
        self.step_size = step_size
        self.attraction_coefficient = attraction_coefficient
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

    def move_grasshoppers(self):
        updated_population = np.copy(self.population)
        for i in range(self.population_size):
            sum_vector = np.zeros(self.dimension)
            for j in range(self.population_size):
                if j != i:
                    distance = np.linalg.norm(self.population[j] - self.population[i])
                    sum_vector += (self.population[j] - self.population[i]) / (distance ** 2)

            step = self.step_size * np.random.uniform() * sum_vector
            attraction = self.attraction_coefficient * sum_vector

            updated_population[i] += step + attraction

        self.population = updated_population

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
            self.move_grasshoppers()

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
                'Step Size': self.step_size,
                'Attraction Coefficient': self.attraction_coefficient,
                'Generations': self.generations
            }
        }
