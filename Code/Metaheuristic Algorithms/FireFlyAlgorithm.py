import numpy as np

class FireflyAlgorithm:
    def __init__(self, objective_func, lb, ub, dim, n_fireflies=40, iterations=100, alpha=0.5, beta_base=1, gamma=1):
        self.objective_func = objective_func
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.n_fireflies = n_fireflies
        self.iterations = iterations
        self.alpha = alpha
        self.beta_base = beta_base
        self.gamma = gamma
        self.population = np.random.uniform(lb, ub, (n_fireflies, dim))
        self.fitness = np.zeros(n_fireflies)
        self.best_cost = float('inf')
        self.best_solution = None
        self.fitness_history = []

    def attractiveness(self, distance):
        return self.beta_base * np.exp(-self.gamma * distance ** 2)

    def move_firefly(self, i, j):
        distance = np.linalg.norm(self.population[i] - self.population[j])
        attractiveness = self.attractiveness(distance)
        random_movement = self.alpha * (np.random.rand(self.dim) - 0.5) * (self.ub - self.lb)
        self.population[i] += attractiveness * (self.population[j] - self.population[i]) + random_movement
        self.population[i] = np.clip(self.population[i], self.lb, self.ub)

    def optimize(self):
        for _ in range(self.iterations):
            self.fitness = np.array([self.objective_func(ind) for ind in self.population])
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if self.fitness[j] < self.fitness[i]:
                        self.move_firefly(i, j)
            if np.min(self.fitness) < self.best_cost:
                self.best_cost = np.min(self.fitness)
                self.best_solution = self.population[np.argmin(self.fitness)].copy()
            self.fitness_history.append(self.best_cost)

        avg_cost = np.mean(self.fitness_history)
        std_dev = np.std(self.fitness_history)
        conv_gen = np.argmin(self.fitness_history)
        return {
            'Best Solution': self.best_solution,
            'Best Cost': self.best_cost,
            'Average Cost': avg_cost,
            'Standard Deviation': std_dev,
            'Convergence Generation': conv_gen,
            'Fitness History': self.fitness_history
        }

if __name__ == "__main__":
    print("This module provides a Firefly Algorithm and is not meant to be run directly.")
