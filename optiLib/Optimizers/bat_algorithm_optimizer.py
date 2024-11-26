from ..Core import Optimizer
import numpy as np
from tqdm import tqdm
import random


class BatAlgorithmOptimizer(Optimizer):
    """
    Оптимизатор на основе алгоритма летучих мышей.
    
    Реализует алгоритм оптимизации, основанный на эхолокации летучих мышей.

    Attributes:
        n_bats (int): Количество летучих мышей
        iterations (int): Количество итераций
        fmin (float): Минимальная частота
        fmax (float): Максимальная частота
        alpha (float): Коэффициент громкости
        gamma (float): Коэффициент импульса
        algo_name (str): Название алгоритма ("BA")
    """

    def __init__(self, problem, n_bats=40, iterations=100,
                 fmin=0, fmax=2, alpha=0.9, gamma=0.9,
                 track_history=True, update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.n_bats = n_bats
        self.iterations = iterations
        self.fmin = fmin
        self.fmax = fmax
        self.alpha = alpha
        self.gamma = gamma
        self.algo_name = "BA"

    def optimize(self):
        """
        Выполняет оптимизацию методом летучих мышей.

        Returns:
            tuple: (best_solution, best_value)
        """
        # Инициализация популяции летучих мышей
        bats = [{'position': self.problem.generate_random_solution(),
                 'velocity': np.zeros(self.problem.n_nodes),
                 'frequency': np.zeros(self.problem.n_nodes),
                 'rate': random.random(),
                 'loudness': random.random()} 
                for _ in range(self.n_bats)]
        
        best_solution = bats[0]['position'].copy()
        best_value = self.problem.evaluate(best_solution)

        for iteration in tqdm(range(self.iterations), desc="Bat Algorithm"):
            for bat in bats:
                # Обновление частоты
                freq = self.fmin + (self.fmax - self.fmin) * random.random()
                
                # Обновление скорости и позиции
                bat['velocity'] = (bat['velocity'] +
                    (bat['position'] - best_solution) * freq)
                new_position = bat['position'] + bat['velocity']
                
                # Локальный поиск
                if random.random() > bat['rate']:
                    new_position = best_solution + \
                        0.001 * np.random.randn(self.problem.n_nodes)
                
                # Округление и ограничение значений
                new_position = np.clip(
                    np.round(new_position), 0, self.problem.n_nodes-1)
                
                # Оценка нового решения
                new_value = self.problem.evaluate(new_position)
                
                # Обновление решения
                if (new_value <= self.problem.evaluate(bat['position']) and
                    random.random() < bat['loudness']):
                    bat['position'] = new_position
                    bat['rate'] = bat['rate'] * (1 - np.exp(-self.gamma * iteration))
                    bat['loudness'] *= self.alpha
                    
                    if new_value < best_value:
                        best_solution = new_position.copy()
                        best_value = new_value
                        self.update_history(iteration, best_solution)
                        
                        if self.first_solution:
                            return best_solution, best_value

        return best_solution, best_value
