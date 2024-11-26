from ..Core import Optimizer
import numpy as np
from tqdm import tqdm
import random

class GeneticOptimizer(Optimizer):
    """
    Оптимизатор на основе генетического алгоритма.
    
    Реализует эволюционный алгоритм оптимизации, использующий механизмы
    селекции, скрещивания и мутации для поиска оптимального решения.

    Attributes:
        population_size (int): Размер популяции
        generations (int): Количество поколений
        mutation_rate (float): Вероятность мутации
        elite_size (int): Количество элитных особей
        algo_name (str): Название алгоритма ("GA" - Genetic Algorithm)

    Args:
        problem: Объект задачи оптимизации
        population_size (int): Размер популяции (по умолчанию 100)
        generations (int): Количество поколений (по умолчанию 100)
        mutation_rate (float): Вероятность мутации (по умолчанию 0.1)
        elite_size (int): Количество элитных особей (по умолчанию 2)
    """

    def __init__(self, problem, population_size=100, generations=100, 
                 mutation_rate=0.1, elite_size=2, track_history=True, 
                 update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.algo_name = "GA"

    def optimize(self):
        """
        Выполняет оптимизацию генетическим алгоритмом.

        Returns:
            tuple: (best_solution, best_value)
        """
        # Инициализация популяции
        population = [self.problem.generate_random_solution() 
                     for _ in range(self.population_size)]
        best_solution = population[0]
        best_value = self.problem.evaluate(best_solution)

        for generation in tqdm(range(self.generations), desc="Genetic Evolution"):
            # Оценка популяции
            fitness = [self.problem.evaluate(ind) for ind in population]
            
            # Обновление лучшего решения
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_value:
                best_solution = population[min_idx].copy()
                best_value = fitness[min_idx]
                self.update_history(generation, best_solution)

            if self.first_solution:
                return best_solution, best_value

            # Селекция и создание нового поколения
            new_population = []
            
            # Сохранение элитных особей
            sorted_indices = np.argsort(fitness)
            new_population.extend([population[i].copy() for i in sorted_indices[:self.elite_size]])
            
            # Создание остальной популяции
            while len(new_population) < self.population_size:
                # Турнирная селекция
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                # Скрещивание
                child = self._crossover(parent1, parent2)
                
                # Мутация
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population

        return best_solution, best_value

    def _tournament_selection(self, population, fitness, tournament_size=3):
        """Турнирная селекция"""
        tournament_idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover(self, parent1, parent2):
        """Одноточечное скрещивание"""
        point = random.randint(1, len(parent1)-1)
        child = np.concatenate([parent1[:point], parent2[point:]])
        return child

    def _mutate(self, solution):
        """Мутация случайного гена"""
        mutation_point = random.randint(0, len(solution)-1)
        solution[mutation_point] = random.randint(0, self.problem.n_nodes-1)
        return solution
