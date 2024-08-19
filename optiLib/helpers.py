# helpers.py
import numpy as np

def check_vector(vector, length):
    if len(vector) != length:
        raise ValueError(f"Вектор должен иметь длину {length}, но имеет длину {len(vector)}.")

def f1(x, problem):
    return x[0]**2 - 4*x[1] + 4

def objective1(x, problem):
    f = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2
    return f

def objective2(x, problem):
    return -np.absolute(np.sin(x[0]) * np.cos(x[1]) * np.exp(np.absolute(1 - (np.sqrt(x[0]**2 + x[1]**2)/np.pi))))


example_constraints = [
    lambda x: x >= 0,
    lambda x: x <= 10
]

example_objective_function = [objective2]

def print_solution(solution, value):
    print(f"Лучшее решение: {solution}")
    print(f"Значение целевой функции: {value}")
