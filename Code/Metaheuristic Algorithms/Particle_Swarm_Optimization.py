# PSO.py
import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(low=bounds[0], high=bounds[1], size=len(bounds[0]))
        self.velocity = np.zeros_like(self.position)
        self.best_position = np.copy(self.position)
        self.best_fitness = float('inf')

    def update_velocity(self, global_best_position, c1, c2, w):
        cognitive_component = c1 * np.random.random(size=len(self.position)) * (self.best_position - self.position)
        social_component = c2 * np.random.random(size=len(self.position)) * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_component + social_component

    def update_position(self, bounds):
        self.position += self.velocity
        # Keep within bounds
        for i in range(len(self.position)):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]

class PSO:
    def __init__(self, objective_func, bounds, num_particles, iterations, c1, c2, w):
        self.objective_func = objective_func
        self.bounds = bounds
        self.num_particles = num_particles
        self.iterations = iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.swarm = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = np.random.uniform(low=bounds[0], high=bounds[1], size=len(bounds[0]))
        self.global_best_fitness = float('inf')

    def run(self):
        fitness_history = []  # Initialize the list to store fitness at each iteration
        for _ in range(self.iterations):
            for particle in self.swarm:
                fitness = self.objective_func(particle.position.reshape(1, -1))[0]
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position

            fitness_history.append(self.global_best_fitness)  # Store the global best fitness

            for particle in self.swarm:
                particle.update_velocity(self.global_best_position, self.c1, self.c2, self.w)
                particle.update_position(self.bounds)

        return self.global_best_position, self.global_best_fitness, fitness_history
