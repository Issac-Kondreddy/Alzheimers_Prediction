import numpy as np

class DifferentialEvolution:
    def __init__(self, objective_func, population_size, chromosome_length, F, CR, generations):
        self.objective_func = objective_func
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.generations = generations
        self.population = None
        self.fitness = None
        self.best_fitness_per_generation = []

    def initialize_population(self):
        self.population = np.random.rand(self.population_size, self.chromosome_length) * 2 - 1  # Adjust the range if needed
        self.fitness = np.zeros(self.population_size)

    def evaluate_population(self):
        for i in range(self.population_size):
            self.fitness[i] = self.objective_func(self.population[i])

    def mutate_and_crossover(self):
        new_population = np.zeros((self.population_size, self.chromosome_length))
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = a + self.F * (b - c)
            cross_points = np.random.rand(self.chromosome_length) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.chromosome_length)] = True
            trial_vector = np.where(cross_points, mutant_vector, self.population[i])
            new_population[i] = trial_vector
        return new_population

    def select_next_generation(self, new_population):
        for i in range(self.population_size):
            new_fitness = self.objective_func(new_population[i])
            if new_fitness < self.fitness[i]:
                self.population[i], self.fitness[i] = new_population[i], new_fitness

    def run(self):
        self.initialize_population()
        self.evaluate_population()
        self.best_fitness_per_generation = []
        self.average_fitness_history = []
        self.std_deviation_history = []

        for generation in range(self.generations):
            new_population = self.mutate_and_crossover()
            self.select_next_generation(new_population)
            self.evaluate_population()

            best_fitness = np.min(self.fitness)
            average_fitness = np.mean(self.fitness)
            std_deviation = np.std(self.fitness)

            self.best_fitness_per_generation.append(best_fitness)
            self.average_fitness_history.append(average_fitness)
            self.std_deviation_history.append(std_deviation)

        best_idx = np.argmin(self.fitness)
        return {
            'Best Solution': self.population[best_idx],
            'Best Fitness': self.fitness[best_idx],
            'Best Fitness Per Generation': self.best_fitness_per_generation,
            'Average Fitness History': self.average_fitness_history,
            'Standard Deviation History': self.std_deviation_history,
            'Parameters': {
                'Population Size': self.population_size,
                'Chromosome Length': self.chromosome_length,
                'F': self.F,
                'CR': self.CR,
                'Generations': self.generations
            }
        }
