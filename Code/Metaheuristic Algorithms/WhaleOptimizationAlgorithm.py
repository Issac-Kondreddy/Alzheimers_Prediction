import numpy as np

class WhaleOptimizationAlgorithm:
    def __init__(self, objective_func, lb, ub, dim, whales_no=30, iterations=100, b=1):
        self.objective_func = objective_func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.whales_no = whales_no
        self.iterations = iterations
        self.b = b
        self.positions = np.random.uniform(self.lb, self.ub, (self.whales_no, self.dim))
        self.best_position = None
        self.best_score = float('inf')
        self.fitness_history = []

    def optimize(self):
        for iteration in range(self.iterations):
            for i in range(self.whales_no):
                fitness = self.objective_func(self.positions[i])
                if fitness < self.best_score:
                    self.best_score = fitness
                    self.best_position = self.positions[i].copy()

            a = 2 - iteration * (2 / self.iterations)  # linearly decreases from 2 to 0

            for i in range(self.whales_no):
                r = np.random.rand()  # [0, 1]
                A = 2 * a * r - a  # [-a, a]
                C = 2 * r
                b = 1  # defines shape of the spiral
                l = (np.random.rand() - 0.5) * 2  # random number in [-1, 1]
                p = np.random.rand()

                for d in range(self.dim):
                    if p < 0.5:
                        if abs(A) < 1:
                            D = abs(C * self.best_position[d] - self.positions[i][d])
                            self.positions[i][d] = self.best_position[d] - A * D
                        else:
                            random_agent_index = np.random.randint(0, self.whales_no)
                            random_agent = self.positions[random_agent_index]
                            D = abs(C * random_agent[d] - self.positions[i][d])
                            self.positions[i][d] = random_agent[d] - A * D
                    else:
                        D = abs(self.best_position[d] - self.positions[i][d])
                        self.positions[i][d] = D * np.exp(b * l) * np.cos(l * 2 * np.pi) + self.best_position[d]

                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            self.fitness_history.append(self.best_score)

        avg_cost = np.mean(self.fitness_history)
        std_dev = np.std(self.fitness_history)
        conv_gen = np.argmin(self.fitness_history)
        return {
            'Best Solution': self.best_position,
            'Best Cost': self.best_score,
            'Average Cost': avg_cost,
            'Standard Deviation': std_dev,
            'Convergence Generation': conv_gen,
            'Fitness History': self.fitness_history
        }

if __name__ == "__main__":
    print("This module provides a Whale Optimization Algorithm and is not meant to be run directly.")
