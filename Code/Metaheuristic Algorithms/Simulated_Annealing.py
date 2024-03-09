import numpy as np

def simulated_annealing(objective_func, bounds, initial_temp, final_temp, alpha, max_iterations):
    current_solution = np.random.uniform(low=bounds[0], high=bounds[1], size=len(bounds[0]))
    current_cost = objective_func(current_solution)
    best_solution = np.copy(current_solution)
    best_cost = current_cost
    temp = initial_temp

    cost_history = [current_cost]  # Keep track of the cost history

    for iteration in range(max_iterations):
        neighbor_solution = current_solution + np.random.uniform(-1, 1, size=current_solution.shape)
        neighbor_solution = np.clip(neighbor_solution, bounds[0], bounds[1])
        neighbor_cost = objective_func(neighbor_solution)

        if neighbor_cost < current_cost or np.random.rand() < np.exp((current_cost - neighbor_cost) / temp):
            current_solution, current_cost = neighbor_solution, neighbor_cost
            cost_history.append(current_cost)  # Update cost history

        if neighbor_cost < best_cost:
            best_solution, best_cost = neighbor_solution, neighbor_cost

        temp *= alpha

        if temp <= final_temp:
            break

    return best_solution, best_cost, cost_history
