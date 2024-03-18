import numpy as np

class GreyWolfOptimizer:
    def __init__(self, objective_func, lb, ub, dim, wolves_no=30, iterations=100):
        self.objective_func = objective_func
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.wolves_no = wolves_no
        self.iterations = iterations
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float("inf")
        self.positions = np.random.uniform(lb, ub, (self.wolves_no, dim))
        self.fitness_history = []

    def optimize(self):
        for t in range(self.iterations):
            for i in range(self.wolves_no):
                fitness = self.objective_func(self.positions[i])
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                self.fitness_history.append(fitness)

            a = 2 - t * (2 / self.iterations)  # Decrease a linearly from 2 to 0

            for i in range(self.wolves_no):
                for j in range(self.dim):
                    A = 2 * a * np.random.rand() - a
                    C = 2 * np.random.rand()
                    D = abs(C * self.alpha_pos[j] - self.positions[i][j])
                    self.positions[i][j] = self.alpha_pos[j] - A * D

            self.positions = np.clip(self.positions, self.lb, self.ub)

        avg_cost = np.mean(self.fitness_history)
        std_dev = np.std(self.fitness_history)
        conv_gen = np.argmin(self.fitness_history)
        return {
            'Best Position': self.alpha_pos,
            'Best Cost': self.alpha_score,
            'Average Cost': avg_cost,
            'Standard Deviation': std_dev,
            'Convergence Generation': conv_gen,
            'Fitness History': self.fitness_history
        }

if __name__ == "__main__":
    print("This module provides a Grey Wolf Optimizer and is not meant to be run directly.")
