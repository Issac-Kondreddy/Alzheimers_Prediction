import numpy as np

class GASSO:
    def __init__(self, objective_func, bounds, population_size, iterations):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.population_size = population_size
        self.iterations = iterations
        self.global_best = np.inf
        self.global_best_position = None
        self.population = self.initialize_population()
        self.fitness_history = []

    def initialize_population(self):
        return np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1],
                                 size=(self.population_size, len(self.bounds)))

    def update_ssa_positions(self):
        # Update the leader's position
        if self.population_size > 0:
            self.population[0] = self.update_leader_position(self.population[0])

        # Update follower positions
        for i in range(1, self.population_size):
            self.population[i] = self.update_follower_position(self.population[i], self.population[i - 1])

    def update_leader_position(self, leader):
        # Simple leader movement towards global best if known; otherwise, random exploration
        if self.global_best_position is not None:
            leader = leader + np.random.random() * (self.global_best_position - leader)
        else:
            leader = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1])
        return np.clip(leader, self.bounds[:, 0], self.bounds[:, 1])

    def update_follower_position(self, follower, previous_follower):
        # Follower movement based on previous follower's position
        follower = follower + np.random.random() * (previous_follower - follower)
        return np.clip(follower, self.bounds[:, 0], self.bounds[:, 1])

    def selection(self):
        # Ensure that selected_indices cannot include an out-of-bounds index
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size // 2, replace=False)
        selected = self.population[selected_indices]
        return selected

    def crossover(self, parents):
        offspring = np.empty((0, len(self.bounds)))
        crossover_point = np.random.randint(1, len(self.bounds)-1)
        for i in range(0, parents.shape[0], 2):
            if i+1 < parents.shape[0]:
                parent1, parent2 = parents[i], parents[i+1]
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
                offspring = np.vstack([offspring, child1, child2])
        return offspring

    def mutation(self, offspring):
        mutation_rate = 0.1
        for i in range(offspring.shape[0]):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(0, len(self.bounds))
                offspring[i, mutation_point] = np.random.uniform(self.bounds[mutation_point, 0], self.bounds[mutation_point, 1])
        return offspring

    def optimize(self):
        for iteration in range(self.iterations):
            # Update global best
            self.evaluate_population()
            self.fitness_history.append(self.global_best)

            if iteration < self.iterations // 2:
                # SSA Phase
                self.update_ssa_positions()
            else:
                # GA Phase
                parents = self.selection()
                offspring = self.crossover(parents)
                # Adjust offspring to match population size if necessary
                while len(offspring) < self.population_size:
                    additional_offspring = self.crossover(parents)
                    offspring = np.vstack((offspring, additional_offspring[:self.population_size - len(offspring)]))
                offspring = offspring[:self.population_size]
                self.population = self.mutation(offspring)

            self.evaluate_population()

        return self.global_best_position, self.global_best, self.fitness_history

    def evaluate_population(self):
        fitness = np.apply_along_axis(self.objective_func, 1, self.population)
        min_fitness_idx = np.argmin(fitness)
        if fitness[min_fitness_idx] < self.global_best:
            self.global_best = fitness[min_fitness_idx]
            self.global_best_position = self.population[min_fitness_idx]

    def calculate_metrics(self):
        average_fitness = np.mean(self.fitness_history)
        std_deviation = np.std(self.fitness_history)
        convergence_threshold = 0.0001
        convergence_generation = next((i for i, fitness in enumerate(self.fitness_history)
                                       if abs(fitness - self.global_best) < convergence_threshold), self.iterations)
        return average_fitness, std_deviation, convergence_generation


