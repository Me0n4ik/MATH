from ..Core import Optimizer
import numpy as np
from tqdm import tqdm
import random

class GreyWolfOptimizer(Optimizer):
    """
    Оптимизатор на основе алгоритма серых волков.
    
    Реализует алгоритм оптимизации, основанный на иерархии и охотничьем поведении
    серых волков. Волки классифицируются как альфа, бета, дельта и омега.

    Attributes:
        n_wolves (int): Количество волков в стае
        iterations (int): Количество итераций
        algo_name (str): Название алгоритма ("GWO")
    """

    def __init__(self, problem, n_wolves=30, iterations=100,
                 track_history=True, update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.n_wolves = n_wolves
        self.iterations = iterations
        self.algo_name = "GWO"

    def optimize(self):
        """
        Выполняет оптимизацию методом серых волков.

        Returns:
            tuple: (best_solution, best_value)
        """
        # Инициализация популяции волков
        wolves = [self.problem.generate_random_solution() 
                 for _ in range(self.n_wolves)]
        fitness = [self.problem.evaluate(w) for w in wolves]

        # Определение альфа, бета и дельта волков
        sorted_wolves = [x for _, x in sorted(zip(fitness, wolves))]
        alpha = sorted_wolves[0].copy()
        beta = sorted_wolves[1].copy()
        delta = sorted_wolves[2].copy()
        
        best_solution = alpha
        best_value = self.problem.evaluate(alpha)

        for iteration in tqdm(range(self.iterations), desc="Grey Wolf"):
            # Обновление параметра a
            a = 2 - iteration * (2/self.iterations)

            for i in range(self.n_wolves):
                # Обновление положения каждого волка
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                D_alpha = abs(C1 * alpha - wolves[i])
                X1 = alpha - A1 * D_alpha

                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                
                D_beta = abs(C2 * beta - wolves[i])
                X2 = beta - A2 * D_beta

                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                D_delta = abs(C3 * delta - wolves[i])
                X3 = delta - A3 * D_delta

                # Обновление позиции волка
                wolves[i] = np.clip(np.round((X1 + X2 + X3)/3), 
                                  0, self.problem.n_nodes-1)
                
                # Оценка нового положения
                new_value = self.problem.evaluate(wolves[i])
                if new_value < best_value:
                    best_solution = wolves[i].copy()
                    best_value = new_value
                    self.update_history(iteration, best_solution)
                    
                    if self.first_solution:
                        return best_solution, best_value

            # Обновление альфа, бета и дельта волков
            fitness = [self.problem.evaluate(w) for w in wolves]
            sorted_wolves = [x for _, x in sorted(zip(fitness, wolves))]
            if self.problem.evaluate(sorted_wolves[0]) < best_value:
                alpha = sorted_wolves[0].copy()
                beta = sorted_wolves[1].copy()
                delta = sorted_wolves[2].copy()

        return best_solution, best_value
