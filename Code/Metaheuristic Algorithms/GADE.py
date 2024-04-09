import numpy as np

class GDE:
    def __init__(self, objective_func, bounds, population_size, iterations):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.iterations = iterations
        self.population = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.population_size, len(self.bounds)))
        self.fitness_history = []

    def evaluate_population(self):
        # Assuming this method evaluates the whole population and finds the best
        # Make sure to append the best fitness of each generation to the fitness_history
        fitnesses = np.apply_along_axis(self.objective_func, 1, self.population)
        self.fitness_history.append(np.min(fitnesses))  # Track fitness history

    def evaluate(self):
        fitness = np.apply_along_axis(self.objective_func, 1, self.population)
        best_idx = np.argmin(fitness)
        return self.population[best_idx], fitness[best_idx]

    def ga_step(self):
        # Tournament Selection
        tournament_size = 3
        selected = np.empty((0, self.population.shape[1]))
        while len(selected) < self.population_size:
            idxs = np.random.randint(0, self.population_size, tournament_size)
            tournament = self.population[idxs]
            fitnesses = np.apply_along_axis(self.objective_func, 1, tournament)
            winner = tournament[np.argmin(fitnesses)]
            selected = np.vstack((selected, winner))

        # Single-point Crossover
        crossover_point = np.random.randint(1, self.population.shape[1])
        children = np.empty((0, self.population.shape[1]))
        for i in range(0, len(selected), 2):
            parent1 = selected[i]
            parent2 = selected[np.random.randint(0, len(selected))] if i + 1 >= len(selected) else selected[i + 1]
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            children = np.vstack((children, child1, child2))

        # Uniform Mutation
        mutation_rate = 0.1
        for i in range(len(children)):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(0, self.population.shape[1])
                children[i, mutation_point] = np.random.uniform(self.bounds[mutation_point, 0],
                                                                self.bounds[mutation_point, 1])

        self.population = children[:self.population_size]

    def de_step(self):
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover rate
        for i, target in enumerate(self.population):
            # Mutation
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + F * (b - c), self.bounds[:, 0], self.bounds[:, 1])

            # Crossover
            trial_vector = np.array(
                [mutant_vector[j] if np.random.rand() < CR else target[j] for j in range(len(target))])

            # Selection
            if self.objective_func(trial_vector) < self.objective_func(target):
                self.population[i] = trial_vector

    def optimize(self):
        for iteration in range(self.iterations):
            if iteration < self.iterations / 2:
                self.ga_step()
            else:
                self.de_step()

            # Evaluate current population and update fitness history
            current_fitnesses = np.apply_along_axis(self.objective_func, 1, self.population)
            best_current_fitness = np.min(current_fitnesses)
            self.fitness_history.append(best_current_fitness)  # Update fitness history directly here

    # Additional methods for GA and DE operations not shown for brevity
    def calculate_metrics(self):
        if not self.fitness_history:
            print("Fitness history is empty. Ensure optimize() has been run.")
            return None, None, None  # or provide default values

        average_fitness = np.mean(self.fitness_history) if self.fitness_history else float('nan')
        std_deviation = np.std(self.fitness_history) if self.fitness_history else float('nan')

        improvement_threshold = 0.0001
        convergence_generation = -1  # Default value indicating no convergence within the threshold
        for i in range(1, len(self.fitness_history)):
            if abs(self.fitness_history[i] - self.fitness_history[i - 1]) < improvement_threshold:
                convergence_generation = i
                break

        return average_fitness, std_deviation, convergence_generation

