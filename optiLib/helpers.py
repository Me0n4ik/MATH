# helpers.py

def check_vector(vector, length):
    if len(vector) != length:
        raise ValueError(f"Вектор должен иметь длину {length}, но имеет длину {len(vector)}.")

def example_objective_function(x):
    return x**2 - 4*x + 4

example_constraints = [
    lambda x: x >= 0,
    lambda x: x <= 10
]

def print_solution(solution, value):
    print(f"Лучшее решение: {solution}")
    print(f"Значение целевой функции: {value}")
