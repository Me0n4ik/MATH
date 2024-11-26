"""
Базовый класс для реализации алгоритмов оптимизации.

Этот модуль предоставляет абстрактный базовый класс для различных алгоритмов оптимизации.
Он включает функциональность для отслеживания истории оптимизации и сохранения результатов.
"""

from copy import copy
from pprint import pprint
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import os


class Optimizer:
    """
    Базовый класс для всех алгоритмов оптимизации.
    
    Attributes:
        problem: Объект, представляющий оптимизационную задачу
        algo_name (str): Название алгоритма
        track_history (bool): Флаг для отслеживания истории оптимизации
        history (list): Список для хранения истории оптимизации
        update_history_coef (int): Коэффициент обновления истории
        update_history_counter (int): Счетчик обновлений истории
        first_solution (bool): Флаг нахождения первого допустимого решения
        b_first_solution (bool): Флаг для отслеживания только первого решения

    Args:
        problem: Объект задачи оптимизации
        track_history (bool): Включить отслеживание истории (по умолчанию True)
        update_history_coef (int): Частота обновления истории (по умолчанию 1000)
        b_first_solution (bool): Отслеживать только первое решение (по умолчанию False)
    """

    def __init__(self, problem, track_history=True, update_history_coef=1000, b_first_solution=False):
        """
        Инициализирует оптимизатор.
        """
        self.problem = problem
        self.algo_name = None
        self.track_history = track_history
        self.history = [] if track_history else None
        self.update_history_coef = update_history_coef
        self.update_history_counter = 0
        self.first_solution = False
        self.b_first_solution = b_first_solution

    def optimize(self):
        """
        Абстрактный метод для реализации алгоритма оптимизации.
        
        Raises:
            NotImplementedError: Если метод не реализован в дочернем классе
        """
        raise NotImplementedError("Метод optimize() должен быть реализован в дочернем классе.")

    def update_history(self, iteration, vector):
        """
        Обновляет историю оптимизации.
        
        Args:
            iteration (int): Номер текущей итерации
            vector (numpy.ndarray): Текущее решение
            
        Notes:
            Метод сохраняет информацию о решениях в соответствии с заданной частотой
            обновления и настройками отслеживания первого решения.
        """
        self.update_history_counter += 1
        
        if self.track_history:
            if self.b_first_solution:
                if not self.first_solution:
                    if self.problem.evaluate(vector) != np.inf:
                        self.first_solution = True
                        self.history.append({
                            'iteration': iteration,
                            'Решение': self.update_history_counter,
                            'Алгоритм': self.algo_name,
                            'vector': vector.copy()
                        })
                else:
                    if self.update_history_counter % self.update_history_coef == 0:
                        self.history.append({
                            'iteration': iteration,
                            'Алгоритм': self.algo_name,
                            'Решение': self.update_history_counter,
                            'vector': vector.copy()
                        })
            else:
                if self.update_history_counter % self.update_history_coef == 0:
                    self.history.append({
                        'iteration': iteration,
                        'Алгоритм': self.algo_name,
                        'Решение': self.update_history_counter,
                        'vector': vector.copy()
                    })

    def save(self, experiment_number):
        """
        Сохраняет результаты оптимизации в Excel файл.
        
        Args:
            experiment_number (int): Номер эксперимента
            
        Notes:
            Метод создает новый файл или дополняет существующий,
            сохраняя все данные с сортировкой по номеру эксперимента и итерации.
        """
        transformed_data = [{
            **d,
            'Эксперимент': experiment_number,
            **self.problem.get_info_save(d['vector'])
        } for d in self.history]
        
        new_df = pd.DataFrame(transformed_data)
        filename = f'./data/{self.problem.name}{self.algo_name}.xlsx'

        if os.path.exists(filename):
            existing_df = pd.read_excel(filename)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df = combined_df.sort_values(['Эксперимент', 'iteration'])
        combined_df.to_excel(filename, index=False)
        print(f"Данные сохранены в файл: {filename}")

    def relod_data(self):
        """
        Сбрасывает историю оптимизации.
        
        Notes:
            Очищает список истории и обнуляет счетчик обновлений.
        """
        self.history = []
        self.update_history_counter = 0
