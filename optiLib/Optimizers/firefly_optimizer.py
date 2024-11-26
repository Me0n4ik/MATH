from ..Core import Optimizer
import numpy as np
from tqdm import tqdm
import random


class FireflyOptimizer(Optimizer):
    """
    Оптимизатор на основе алгоритма светлячков.
    
    Реализует алгоритм оптимизации, основанный на поведении светлячков,
    где более яркие особи привлекают менее ярких.

    Attributes:
        n_fireflies (int): Количество светлячков
        iterations (int): Количество итераций
        alpha (float): Коэффициент случайности
        beta0 (float): Начальная привлекательность
        gamma (float): Коэффициент поглощения света
        algo_name (str): Название алгоритма ("FA")
    """

    def __init__(self, problem, n_fireflies=50, iterations=100,
                 alpha=0.5, beta0=1.0, gamma=1.0, track_history=True,
                 update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.n_fireflies = n_fireflies
        self.iterations = iterations
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.algo_name = "FA"

    def optimize(self):
        """
        Выполняет оптимизацию методом светлячков.

        Returns:
            tuple: (best_solution, best_value)
        """
        # Инициализация популяции светлячков
        fireflies = [self.problem.generate_random_solution() 
                    for _ in range(self.n_fireflies)]
        brightness = [self.problem.evaluate(f) for f in fireflies]
        
        best_idx = np.argmin(brightness)
        best_solution = fireflies[best_idx].copy()
        best_value = brightness[best_idx]

        for iteration in tqdm(range(self.iterations), desc="Firefly Algorithm"):
            # Движение светлячков
            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if brightness[j] < brightness[i]:
                        # Вычисление расстояния
                        r = np.sqrt(np.sum((fireflies[i] - fireflies[j])**2))
                        # Вычисление привлекательности
                        beta = self.beta0 * np.exp(-self.gamma * r**2)
                        
                        # Движение светлячка
                        fireflies[i] = fireflies[i] + beta * (fireflies[j] - fireflies[i]) + \
                            self.alpha * (random.random() - 0.5)
                        
                        # Округление и ограничение значений
                        fireflies[i] = np.clip(
                            np.round(fireflies[i]), 0, self.problem.n_nodes-1)
                        
                        # Обновление яркости
                        brightness[i] = self.problem.evaluate(fireflies[i])
                        
                        if brightness[i] < best_value:
                            best_solution = fireflies[i].copy()
                            best_value = brightness[i]
                            self.update_history(iteration, best_solution)
                            
                            if self.first_solution:
                                return best_solution, best_value

        return best_solution, best_value
