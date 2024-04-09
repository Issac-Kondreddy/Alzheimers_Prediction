import numpy as np

class DESSA:
    def __init__(self, objective_func, bounds, population_size, iterations):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.iterations = iterations
        self.population = self.initialize_population()
        self.fitness_history = []
        self.global_best = None
        self.global_best_fitness = np.inf

    def initialize_population(self):
        return np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.population_size, len(self.bounds)))

    def evaluate_population(self):
        fitnesses = np.apply_along_axis(self.objective_func, 1, self.population)
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < self.global_best_fitness:
            self.global_best_fitness = fitnesses[min_idx]
            self.global_best = self.population[min_idx]
        self.fitness_history.append(self.global_best_fitness)

    def differential_evolution_step(self):
        F = 0.5  # Mutation factor
        CR = 0.7  # Crossover probability

        for i in range(self.population_size):
            # Mutation
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), self.bounds[:, 0], self.bounds[:, 1])

            # Crossover
            cross_points = np.random.rand(len(self.bounds)) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, len(self.bounds))] = True
            trial = np.where(cross_points, mutant, self.population[i])

            # Selection
            if self.objective_func(trial) < self.objective_func(self.population[i]):
                self.population[i] = trial

    def salp_swarm_step(self):
        # Update the leader (first salp) towards the best solution found so far
        if self.global_best is not None:
            self.population[0] += 0.5 * (self.global_best - self.population[0])
            self.population[0] = np.clip(self.population[0], self.bounds[:, 0], self.bounds[:, 1])

        # Update follower salps
        for i in range(1, self.population_size):
            self.population[i] += 0.5 * (self.population[i - 1] - self.population[i])
            self.population[i] = np.clip(self.population[i], self.bounds[:, 0], self.bounds[:, 1])
    def optimize(self):
        for iteration in range(self.iterations):
            self.evaluate_population()
            if iteration < self.iterations / 2:
                self.differential_evolution_step()
            else:
                self.salp_swarm_step()
        return self.global_best, self.global_best_fitness, self.fitness_history

    def calculate_metrics(self):
        average_fitness = np.mean(self.fitness_history)
        std_deviation = np.std(self.fitness_history)
        convergence_threshold = 0.0001
        convergence_generation = next((i for i, fitness in enumerate(self.fitness_history) if abs(fitness - self.global_best_fitness) < convergence_threshold), -1)
        return average_fitness, std_deviation, convergence_generation
