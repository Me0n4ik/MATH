from ..Core import Optimizer
import numpy as np
from tqdm import tqdm
import random


class DifferentialEvolutionOptimizer(Optimizer):
    """
    Оптимизатор на основе алгоритма дифференциальной эволюции.
    
    Реализует эволюционный алгоритм оптимизации, использующий разности между
    векторами решений для создания новых кандидатов.

    Attributes:
        population_size (int): Размер популяции
        generations (int): Количество поколений
        F (float): Масштабный коэффициент мутации
        CR (float): Вероятность кроссовера
        algo_name (str): Название алгоритма ("DE")
    """

    def __init__(self, problem, population_size=50, generations=100,
                 F=0.8, CR=0.7, track_history=True, update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        self.algo_name = "DE"

    def optimize(self):
        """
        Выполняет оптимизацию методом дифференциальной эволюции.

        Returns:
            tuple: (best_solution, best_value)
        """
        # Инициализация популяции
        population = [self.problem.generate_random_solution() 
                     for _ in range(self.population_size)]
        fitness = [self.problem.evaluate(ind) for ind in population]
        
        best_index = np.argmin(fitness)
        best_solution = population[best_index].copy()
        best_value = fitness[best_index]

        for generation in tqdm(range(self.generations), desc="Differential Evolution"):
            for i in range(self.population_size):
                # Выбор трёх случайных индексов, отличных от i
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = random.sample(candidates, 3)
                
                # Мутация
                mutant = population[a] + self.F * (population[b] - population[c])
                
                # Кроссовер
                trial = np.zeros_like(population[i])
                for j in range(self.problem.n_nodes):
                    if random.random() < self.CR:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = population[i][j]
                
                # Округление и ограничение значений
                trial = np.clip(np.round(trial), 0, self.problem.n_nodes-1)
                
                # Оценка пробного решения
                trial_value = self.problem.evaluate(trial)
                
                # Отбор
                if trial_value < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_value
                    
                    if trial_value < best_value:
                        best_solution = trial.copy()
                        best_value = trial_value
                        self.update_history(generation, best_solution)
                        
                        if self.first_solution:
                            return best_solution, best_value

        return best_solution, best_value
