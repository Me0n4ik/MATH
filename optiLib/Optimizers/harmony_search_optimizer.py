from ..Core import Optimizer
import numpy as np
from tqdm import tqdm
import random


class HarmonySearchOptimizer(Optimizer):
    """
    Оптимизатор на основе алгоритма поиска гармонии.
    
    Реализует алгоритм оптимизации, основанный на музыкальной импровизации,
    где каждое решение представляет собой "гармонию".

    Attributes:
        n_harmonies (int): Размер памяти гармоний
        iterations (int): Количество итераций
        hmcr (float): Вероятность выбора из памяти гармоний
        par (float): Вероятность настройки тона
        bw (float): Ширина полосы настройки
        algo_name (str): Название алгоритма ("HS")
    """

    def __init__(self, problem, n_harmonies=30, iterations=100,
                 hmcr=0.9, par=0.3, bw=0.02, track_history=True,
                 update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.n_harmonies = n_harmonies
        self.iterations = iterations
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.algo_name = "HS"

    def optimize(self):
        """
        Выполняет оптимизацию методом поиска гармонии.

        Returns:
            tuple: (best_solution, best_value)
        """
        # Инициализация памяти гармоний
        harmony_memory = [self.problem.generate_random_solution() 
                         for _ in range(self.n_harmonies)]
        fitness = [self.problem.evaluate(h) for h in harmony_memory]
        
        best_solution = harmony_memory[np.argmin(fitness)].copy()
        best_value = min(fitness)

        for iteration in tqdm(range(self.iterations), desc="Harmony Search"):
            # Создание новой гармонии
            new_harmony = np.zeros(self.problem.n_nodes)
            
            for i in range(self.problem.n_nodes):
                if random.random() < self.hmcr:
                    # Выбор из памяти гармоний
                    memory_index = random.randint(0, self.n_harmonies-1)
                    new_harmony[i] = harmony_memory[memory_index][i]
                    
                    # Настройка тона
                    if random.random() < self.par:
                        new_harmony[i] += random.uniform(-self.bw, self.bw)
                else:
                    # Случайная генерация
                    new_harmony[i] = random.randint(0, self.problem.n_nodes-1)
            
            # Округление и ограничение значений
            new_harmony = np.clip(np.round(new_harmony), 
                                0, self.problem.n_nodes-1)
            
            # Оценка новой гармонии
            new_value = self.problem.evaluate(new_harmony)
            
            # Обновление памяти гармоний
            worst_index = np.argmax(fitness)
            if new_value < fitness[worst_index]:
                harmony_memory[worst_index] = new_harmony
                fitness[worst_index] = new_value
                
                if new_value < best_value:
                    best_solution = new_harmony.copy()
                    best_value = new_value
                    self.update_history(iteration, best_solution)
                    
                    if self.first_solution:
                        return best_solution, best_value

        return best_solution, best_value
