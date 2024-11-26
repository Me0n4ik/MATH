from ..Core import Optimizer
import numpy as np
from tqdm import tqdm
import random

class SimulatedAnnealingOptimizer(Optimizer):
    """
    Оптимизатор на основе алгоритма имитации отжига.
    
    Реализует алгоритм оптимизации, основанный на физическом процессе отжига,
    позволяющий избегать локальных минимумов.

    Attributes:
        initial_temp (float): Начальная температура
        final_temp (float): Конечная температура
        cooling_rate (float): Коэффициент охлаждения
        iterations (int): Количество итераций
        algo_name (str): Название алгоритма ("SA" - Simulated Annealing)
    """

    def __init__(self, problem, initial_temp=100.0, final_temp=1e-4,
                 cooling_rate=0.95, iterations=1000, track_history=True,
                 update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.algo_name = "SA"

    def optimize(self):
        """
        Выполняет оптимизацию методом имитации отжига.

        Returns:
            tuple: (best_solution, best_value)
        """
        current_solution = self.problem.generate_random_solution()
        current_value = self.problem.evaluate(current_solution)
        
        best_solution = current_solution.copy()
        best_value = current_value
        
        temperature = self.initial_temp
        
        for iteration in tqdm(range(self.iterations), desc="Simulated Annealing"):
            # Генерация нового решения
            neighbor = self._get_neighbor(current_solution)
            neighbor_value = self.problem.evaluate(neighbor)
            
            # Вычисление дельты
            delta = neighbor_value - current_value
            
            # Принятие решения о переходе
            if (delta < 0 or 
                random.random() < np.exp(-delta / temperature)):
                current_solution = neighbor
                current_value = neighbor_value
                
                # Обновление лучшего решения
                if current_value < best_value:
                    best_solution = current_solution.copy()
                    best_value = current_value
                    self.update_history(iteration, best_solution)
                    
                    if self.first_solution:
                        return best_solution, best_value
            
            # Охлаждение
            temperature *= self.cooling_rate
            if temperature < self.final_temp:
                break

        return best_solution, best_value

    def _get_neighbor(self, solution):
        """Генерация соседнего решения"""
        neighbor = solution.copy()
        idx = random.randint(0, len(solution)-1)
        neighbor[idx] = random.randint(0, self.problem.n_nodes-1)
        return neighbor
