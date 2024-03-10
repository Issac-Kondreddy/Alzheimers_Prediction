import numpy as np

class TabuSearch:
    def __init__(self, objective_func, bounds, num_iterations, tabu_tenure, num_neighbors):
        self.objective_func = objective_func
        self.bounds = np.array(bounds)
        self.num_iterations = num_iterations
        self.tabu_tenure = tabu_tenure
        self.num_neighbors = num_neighbors
        self.best_cost = float('inf')
        self.best_position = np.random.uniform(low=self.bounds[:,0], high=self.bounds[:,1], size=(len(bounds)))

    def generate_neighbors(self, current_position):
        neighbors = []
        for _ in range(self.num_neighbors):
            # Perturb the current position by a small amount within the bounds
            neighbor = current_position + np.random.uniform(-1, 1, size=current_position.shape) * 0.1 * (self.bounds[:,1] - self.bounds[:,0])
            # Ensure the neighbor stays within bounds
            neighbor = np.clip(neighbor, self.bounds[:,0], self.bounds[:,1])
            neighbors.append(neighbor)
        return np.array(neighbors)

    def optimize(self):
        current_position = self.best_position.copy()
        current_cost = self.objective_func(current_position.reshape(1, -1))
        tabu_list = [current_position.tolist()]
        fitness_history = [current_cost]

        for it in range(self.num_iterations):
            neighbors = self.generate_neighbors(current_position)
            neighbor_costs = np.array([self.objective_func(neighbor.reshape(1, -1)) for neighbor in neighbors])

            # Exclude tabu solutions
            non_tabu_indices = [i for i, neighbor in enumerate(neighbors) if neighbor.tolist() not in tabu_list]

            if non_tabu_indices:  # If there are non-tabu neighbors
                # Find the best among non-tabu neighbors
                best_non_tabu_index = non_tabu_indices[np.argmin(neighbor_costs[non_tabu_indices])]
                best_non_tabu_neighbor = neighbors[best_non_tabu_index]
                best_non_tabu_cost = neighbor_costs[best_non_tabu_index]

                # Move to the best non-tabu neighbor
                current_position = best_non_tabu_neighbor
                current_cost = best_non_tabu_cost

                # Update the best solution found so far
                if current_cost < self.best_cost:
                    self.best_cost = current_cost
                    self.best_position = current_position

                # Update the tabu list
                tabu_list.append(current_position.tolist())
                if len(tabu_list) > self.tabu_tenure:
                    tabu_list.pop(0)

            fitness_history.append(self.best_cost)

        return self.best_position, self.best_cost, fitness_history

