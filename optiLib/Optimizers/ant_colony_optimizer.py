from ..Core import Optimizer
import numpy as np
from tqdm import tqdm
import random



class AntColonyOptimizer(Optimizer):
    """
    Оптимизатор на основе алгоритма муравьиной колонии.
    
    Реализует алгоритм оптимизации, основанный на поведении муравьев
    при поиске пути к источнику пищи.

    Attributes:
        n_ants (int): Количество муравьев
        iterations (int): Количество итераций
        alpha (float): Важность феромона
        beta (float): Важность эвристической информации
        evaporation (float): Скорость испарения феромона
        Q (float): Количество откладываемого феромона
    """

    def __init__(self, problem, n_ants=50, iterations=100, alpha=1.0,
                 beta=2.0, evaporation=0.1, Q=1.0, track_history=True,
                 update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q
        self.algo_name = "ACO"

    def optimize(self):
        """
        Выполняет оптимизацию методом муравьиной колонии.

        Returns:
            tuple: (best_solution, best_value)
        """
        # Инициализация феромонов
        n_nodes = self.problem.n_nodes
        pheromone = np.ones((n_nodes, n_nodes)) / n_nodes
        
        # Вычисление эвристической информации
        heuristic = self._calculate_heuristic()
        
        best_solution = None
        best_value = np.inf

        for iteration in tqdm(range(self.iterations), desc="Ant Colony"):
            # Построение решений муравьями
            solutions = []
            for _ in range(self.n_ants):
                solution = self._construct_solution(pheromone, heuristic)
                solutions.append(solution)
                
                # Оценка решения
                value = self.problem.evaluate(solution)
                if value < best_value:
                    best_solution = solution.copy()
                    best_value = value
                    self.update_history(iteration, best_solution)
                    
                    if self.first_solution:
                        return best_solution, best_value
            
            # Обновление феромонов
            self._update_pheromone(pheromone, solutions)

        return best_solution, best_value

    def _calculate_heuristic(self):
        """Вычисление эвристической информации"""
        n_nodes = self.problem.n_nodes
        heuristic = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # Используем обратное расстояние как эвристику
                    distance = np.sqrt(np.sum((self.problem.coordinates[i] - 
                                             self.problem.coordinates[j])**2))
                    heuristic[i,j] = 1.0 / (distance + 1e-10)
        return heuristic

    def _construct_solution(self, pheromone, heuristic):
        """Построение решения одним муравьем"""
        current = self.problem.source
        path = [current]
        
        while current != self.problem.destination:
            # Вычисление вероятностей перехода
            probabilities = self._calculate_probabilities(
                current, path, pheromone, heuristic)
            
            # Выбор следующего узла
            next_node = np.random.choice(
                range(self.problem.n_nodes), p=probabilities)
            path.append(next_node)
            current = next_node
            
            if len(path) >= self.problem.n_nodes:
                break
        
        return np.array(path)

    def _calculate_probabilities(self, current, path, pheromone, heuristic):
        """Вычисление вероятностей перехода"""
        probabilities = np.zeros(self.problem.n_nodes)
        for next_node in range(self.problem.n_nodes):
            if next_node not in path:
                probabilities[next_node] = (
                    pheromone[current, next_node]**self.alpha * 
                    heuristic[current, next_node]**self.beta
                )
        return probabilities / (np.sum(probabilities) + 1e-10)

    def _update_pheromone(self, pheromone, solutions):
        """Обновление феромонов"""
        # Испарение
        pheromone *= (1 - self.evaporation)
        
        # Добавление нового феромона
        for solution in solutions:
            value = self.problem.evaluate(solution)
            if value != np.inf:
                for i in range(len(solution)-1):
                    pheromone[solution[i], solution[i+1]] += self.Q / value
