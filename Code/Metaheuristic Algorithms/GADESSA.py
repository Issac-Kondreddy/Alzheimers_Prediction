import numpy as np
class DEGASSA:
    def __init__(self, objective_func, bounds, population_size, iterations):
        self.objective_func = objective_func  # The function to optimize
        self.bounds = np.array(bounds)  # The bounds for each dimension of the search space
        self.population_size = population_size  # The size of the population
        self.iterations = iterations  # Total number of iterations for the optimization process
        self.dimension = len(bounds)  # Number of dimensions in the search space
        self.population = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(self.population_size, self.dimension))  # Initialize population
        self.fitness_history = []  # To track the fitness of the best solution over time

    def evaluate(self):
        # Evaluate the entire population to find the best solution and its fitness
        fitnesses = np.apply_along_axis(self.objective_func, 1, self.population)
        best_idx = np.argmin(fitnesses)
        best_idx = min(best_idx, self.population.shape[0] - 1)
        # Assuming a minimization problem
        print(f"Population size: {self.population.shape[0]}, Best index: {best_idx}")  # Debugging print
        best_solution = self.population[best_idx]
        best_fitness = fitnesses[best_idx]
        return best_solution, best_fitness

    def optimize(self):
        print("Starting optimization...")
        for iteration in range(self.iterations):
            print(f"Iteration {iteration + 1}/{self.iterations}")
            if iteration < self.iterations / 3:
                print("Applying GA step")
                self.ga_step()
            elif iteration < 2 * self.iterations / 3:
                print("Applying DE step")
                self.de_step()
            else:
                print("Applying SSA step")
                self.ssa_step()

            _, best_fitness = self.evaluate()
            self.fitness_history.append(best_fitness)
            print(f"Best fitness so far: {best_fitness}")

        print("Optimization complete")
        return self.evaluate()
    # Implement ga_step, de_step, ssa_step, and evaluate methods
    def ga_step(self):
        # Tournament Selection
        new_population = []
        for _ in range(self.population_size):
            participants = self.population[np.random.randint(0, self.population_size, 3)]
            fitnesses = np.apply_along_axis(self.objective_func, 1, participants)
            winner = participants[np.argmin(fitnesses)]
            new_population.append(winner)

        # Crossover
        children = []
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size:
                parent1, parent2 = new_population[i], new_population[i + 1]
                crossover_point = np.random.randint(1, self.dimension)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                children.extend([child1, child2])
            else:
                # If population size is odd, carry the last individual
                children.append(new_population[i])

        # Mutation
        for child in children:
            if np.random.rand() < 0.1:  # Mutation probability
                mutation_point = np.random.randint(self.dimension)
                child[mutation_point] = np.random.uniform(self.bounds[mutation_point, 0],
                                                          self.bounds[mutation_point, 1])

        self.population = np.array(children)

    def de_step(self):
        F = 0.8  # Mutation factor
        CR = 0.7  # Crossover rate
        new_population = np.copy(self.population)
        for i in range(self.population_size):
            target = self.population[i]
            indices = [index for index in range(self.population_size) if index != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), self.bounds[:, 0], self.bounds[:, 1])  # Mutation

            # Crossover
            cross_points = np.random.rand(self.dimension) < CR
            trial = np.where(cross_points, mutant, target)

            # Selection
            if self.objective_func(trial) < self.objective_func(target):
                new_population[i] = trial

        self.population = new_population

    def ssa_step(self):
        # Sort the population based on fitness
        fitnesses = np.apply_along_axis(self.objective_func, 1, self.population)
        sorted_indices = np.argsort(fitnesses)
        self.population = self.population[sorted_indices]

        # Update positions based on the Salp Swarm Algorithm
        # Leader update
        for i in range(1, self.population_size // 2):
            c1 = np.random.random(self.dimension)  # Random weights for leader's influence
            c2 = np.random.random(self.dimension)  # Random weights for position influence
            # The leader moves towards the best solution
            self.population[i] = c1 * self.population[0] + c2 * (self.population[i] - self.population[0])

        # Follower update
        for i in range(self.population_size // 2, self.population_size):
            c1 = np.random.random(self.dimension)  # Random weights for the previous salp's influence
            # Followers follow their predecessor
            self.population[i] = c1 * self.population[i - 1]

        # Ensure all solutions are within bounds
        self.population = np.clip(self.population, self.bounds[:, 0], self.bounds[:, 1])

    def calculate_metrics(self):
        if not self.fitness_history:
            return None, None, None, None  # Handle case with no fitness history

        # Best fitness is the minimum since we assume a minimization problem
        best_fitness = np.min(self.fitness_history)

        # Average fitness and standard deviation over all recorded best fitnesses
        average_fitness = np.mean(self.fitness_history)
        std_deviation = np.std(self.fitness_history)

        # Convergence generation: first generation where the change in best fitness falls below a threshold
        # This simplistic approach assumes convergence criteria based on changes in best fitness
        convergence_generation = None
        for i in range(1, len(self.fitness_history)):
            if abs(self.fitness_history[i] - self.fitness_history[
                i - 1]) < 0.0001:  # Threshold for detecting convergence
                convergence_generation = i
                break

        return best_fitness, average_fitness, std_deviation, convergence_generation




