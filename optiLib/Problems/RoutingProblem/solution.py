import numpy as np

class Solution:
    """Класс для представления решения маршрутизации"""
    def __init__(self, path, problem):
        self.path = np.array(path)
        self.problem = problem
        self.energy = None
        self.reliability = None
        self.evaluate()
    
    def evaluate(self):
        """Вычисление характеристик решения"""
        self.energy = self.problem.energy_consumption(self.path, self.problem)
        self.reliability = self.problem.connection_probability(self.path, self.problem)
        
    def is_valid(self):
        """Проверка корректности решения"""
        return self.problem.check_constraints(self.path)
    
    def copy(self):
        """Создание копии решения"""
        return Solution(self.path.copy(), self.problem)
    
    def __str__(self):
        return f"Path: {self.path}, Energy: {self.energy:.2f}, Reliability: {self.reliability:.2f}"
