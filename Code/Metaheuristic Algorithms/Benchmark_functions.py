import numpy as np

def sphere(x):
    # Ensure x is a 2D array
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return np.sum(x**2, axis=1)

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    # Ensure x is a 2D array; if not, reshape it.
    if x.ndim == 1:
        x = np.reshape(x, (1, -1))
    
    d = x.shape[1]
    sum_sq_term = -0.2 * np.sqrt(np.sum(x ** 2, axis=1) / d)
    cos_term = np.sum(np.cos(c * x), axis=1) / d
    return -a * np.exp(sum_sq_term) - np.exp(cos_term) + a + np.exp(1)


def rastrigin(x):
    # Ensure x is a 2D array; if not, reshape it
    if x.ndim == 1:
        x = np.reshape(x, (1, -1))
    
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10, axis=1)



def rosenbrock(x):
    # Ensure x is a 2D array; if not, reshape it.
    if x.ndim == 1:
        x = np.reshape(x, (1, -1))
    
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0 + (1 - x[:, :-1]) ** 2.0, axis=1)

