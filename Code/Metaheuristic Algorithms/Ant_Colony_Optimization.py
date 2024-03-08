import numpy as np


class Ant:
    def __init__(self, bounds):
        self.position = np.random.uniform(low=bounds[0], high=bounds[1], size=len(bounds[0]))
        self.cost = float('inf')


class ACO:
    def __init__(self, objective_func, bounds, num_ants, iterations, pheromone_evaporation_coeff, pheromone_constant,
                 alpha, beta):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_ants = num_ants
        self.iterations = iterations
        self.pheromone_evaporation_coeff = pheromone_evaporation_coeff
        self.pheromone_constant = pheromone_constant
        self.alpha = alpha  # Influence of pheromone
        self.beta = beta  # Influence of heuristic information
        self.ants = [Ant(bounds) for _ in range(num_ants)]
        self.global_best_cost = float('inf')
        self.global_best_position = np.array([0.0] * len(bounds[0]))
        self.pheromone = np.ones((num_ants, len(bounds[0])))  # Pheromone matrix

    def update_pheromones(self):
        for ant in self.ants:
            for i in range(len(self.bounds[0])):
                # Update pheromone trail levels
                self.pheromone[:, i] *= (1 - self.pheromone_evaporation_coeff)
                self.pheromone[:, i] += self.pheromone_constant / ant.cost

    def optimize(self):
        fitness_history = []
        for _ in range(self.iterations):
            for ant in self.ants:
                # Update position based on pheromones and heuristic (random in this simplified version)
                for i in range(len(self.bounds[0])):
                    ant.position[i] = np.random.uniform(self.bounds[0][i], self.bounds[1][i])

                # Evaluate fitness
                ant.cost = self.objective_func(ant.position.reshape(1, -1))[0]

                # Update global best if found a new best
                if ant.cost < self.global_best_cost:
                    self.global_best_cost = ant.cost
                    self.global_best_position = ant.position

            self.update_pheromones()
            fitness_history.append(self.global_best_cost)

        return self.global_best_position, self.global_best_cost, fitness_history
