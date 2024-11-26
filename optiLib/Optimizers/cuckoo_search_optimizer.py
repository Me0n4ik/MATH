from ..Core import Optimizer
import numpy as np
from tqdm import tqdm
import random
from scipy.special import gamma


class CuckooSearchOptimizer(Optimizer):
    """
    Оптимизатор на основе алгоритма поиска кукушки.
    
    Реализует алгоритм оптимизации, основанный на поведении кукушек
    при откладывании яиц в гнезда других птиц.

    Attributes:
        n_nests (int): Количество гнезд
        iterations (int): Количество итераций
        pa (float): Вероятность обнаружения кукушки
        algo_name (str): Название алгоритма ("CS")
    """

    def __init__(self, problem, n_nests=25, iterations=100,
                 pa=0.25, track_history=True, update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.n_nests = n_nests
        self.iterations = iterations
        self.pa = pa
        self.algo_name = "CS"

    def optimize(self):
        """
        Выполняет оптимизацию методом поиска кукушки.

        Returns:
            tuple: (best_solution, best_value)
        """
        # Инициализация гнезд
        nests = [self.problem.generate_random_solution() 
                for _ in range(self.n_nests)]
        fitness = [self.problem.evaluate(nest) for nest in nests]
        
        best_idx = np.argmin(fitness)
        best_solution = nests[best_idx].copy()
        best_value = fitness[best_idx]

        for iteration in tqdm(range(self.iterations), desc="Cuckoo Search"):
            # Получение нового решения через полёт Леви
            cuckoo = self._levy_flight(best_solution)
            
            # Случайное гнездо для замены
            j = random.randint(0, self.n_nests-1)
            if self.problem.evaluate(cuckoo) < fitness[j]:
                nests[j] = cuckoo
                fitness[j] = self.problem.evaluate(cuckoo)
                
                if fitness[j] < best_value:
                    best_solution = cuckoo.copy()
                    best_value = fitness[j]
                    self.update_history(iteration, best_solution)
                    
                    if self.first_solution:
                        return best_solution, best_value
            
            # Обнаружение и замена худших гнезд
            self._abandon_worst_nests(nests, fitness)

        return best_solution, best_value

    def _levy_flight(self, current):
        """Генерация нового решения через полёт Леви"""
        beta = 1.5
        sigma = (gamma(1+beta) * np.sin(np.pi*beta/2) / 
                (gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
        u = np.random.normal(0, sigma, self.problem.n_nodes)
        v = np.random.normal(0, 1, self.problem.n_nodes)
        step = u / abs(v)**(1/beta)
        
        new_solution = current + step
        return np.clip(np.round(new_solution), 0, self.problem.n_nodes-1)

    def _abandon_worst_nests(self, nests, fitness):
        """Замена худших гнезд новыми"""
        for i in range(self.n_nests):
            if random.random() < self.pa:
                nests[i] = self.problem.generate_random_solution()
                fitness[i] = self.problem.evaluate(nests[i])
