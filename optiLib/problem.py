# opti/problem.py

import numpy as np
import networkx as nx
from dataclasses import dataclass
import functools

class OptimizationProblem:
    def __init__(self, f_objective, f_constraints=None, bounds=None, dtype = int, len = 10, name = "Problem 1"):
        """
        :param initial_vector: Исходный вектор, который мы оптимизируем.
        :param f_objective: Список целевых функций, которые мы хотим оптимизировать.
        :param f_constraints: Список функций-ограничений, которым должен соответствовать вектор.
        :param v_constraints: Список ограничений, которым должен соответствовать вектор.
        """
        self.name = name
        self.f_objective = f_objective
        self.f_constraints = f_constraints if f_constraints is not None else []
        self.vector_length = None
        self.bounds = np.array(bounds) if bounds is not None else None
        self.dtype = dtype
        self.vector_length = len

    def generate_random_solution(self):
        if np.random.random() < 0.7:
            net_nodes = self.network_graph.graph.number_of_nodes()
            tasks = self.task_graph.graph.number_of_nodes()
            num_to_select = np.random.randint(1, net_nodes + 1)
            nods_to_select = np.random.choice([i for i in range(net_nodes)], size=num_to_select, replace=False)
            return np.random.choice(nods_to_select, size=tasks)
        else:
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds[:, 0], self.bounds[:, 1]
                return np.random.uniform(lower_bounds, upper_bounds, self.vector_length).astype(self.dtype)
            else:
                return np.zeros(self.vector_length, dtype=self.dtype)

    def get_info(self, vector=None):
        return {**{f.__name__:f(vector, self) for f in self.f_objective}, **{f.__name__:f(vector, self) for f in self.f_constraints}, 'Свертка': self.evaluate(vector)}

    def evaluate_objectives(self, vector=None):
        """Оценивает все целевые функции на заданном векторе. Возвращает список значений.""" 
        return [f(vector, self) for f in self.f_objective]
    
    def convolution_evaluate_objectives(self, vector=None):
        """Выдает свертку целевых функций по заданому вветору"""
        return np.prod([f(vector, self) for f in self.f_objective])

    def expanded_constraints(self, vector=None):
        """Проверяет выполнение функций-условий по заданому вектору. Возращает булевый вектор."""
        return [c(vector, self) for c in self.f_constraints]

    def check_constraints(self, vector=None):
        """Проверяет, выполняются ли все ограничения на заданном векторе."""
        return all(c(vector, self) for c in self.f_constraints)

    def evaluate(self, solution):
        solution = self.constrain_elements(solution)
        if self.is_feasible(solution):
            return self.convolution_evaluate_objectives(solution)
        return np.inf
    
    def constrain_elements(self, vector):
        """Приводит элементы вектора к ближайшим допустимым значениям в соответствии с ограничениями."""
        vector = np.array(vector).astype(self.dtype)

        if self.bounds is not None:
            lower_bounds, upper_bounds = self.bounds[:, 0], self.bounds[:, 1]
            return np.clip(vector, lower_bounds, upper_bounds)
        return vector

    def is_feasible(self, solution):
        if len(solution) != self.vector_length:
            return False
        return self.check_constraints(solution)

class IntegerOptimizationProblem(OptimizationProblem):
    def __init__(self, f_objective, f_constraints=None, bounds=None, len = 10):
        super().__init__(f_objective, f_constraints=None, bounds=None, dtype = int, len = 10)

    def evaluate(self, solution):
        return super().evaluate(solution)


@dataclass
class NetNode:
    p: int = 0
    e0: int = 1
    eMax: int = 1

class NetGraph:
    def __init__(self, martx, net_power = (100, 2500), net_power_arr = None, e0 = (0,70), emax = (70,100)) -> None:
        self.graph = nx.Graph(np.array(martx))
        if net_power_arr is not None:
            self.nodes = [NetNode(net_power_arr[i], np.random.randint(e0[0], e0[1]), np.random.randint(emax[0], emax[1])) for i in range(self.graph.number_of_nodes())]
        else:
            self.nodes = [NetNode(np.random.randint(net_power[0], net_power[1]), np.random.randint(e0[0], e0[1]), np.random.randint(emax[0], emax[1])) \
                      for _ in range(self.graph.number_of_nodes())]
        
    def __str__(self) -> str:
        return '\n'.join([f'{node}' for node in self.nodes])
    
    def print(self):
        pos = nx.spring_layout(self.graph, seed=100)
        nx.draw(self.graph, pos, with_labels=True, font_color='white')


@dataclass
class TaskNode:
    w: int = 0

class TaskGraph:
    def __init__(self, martx, w = (100,600), w_arr = None) -> None:
        self.graph = nx.DiGraph(np.array(martx))
        if w_arr is not None:
            self.operations = [TaskNode(w_arr[i]) for i in range(self.graph.number_of_nodes())]
        else:
            self.operations = [TaskNode(np.random.randint(w[0], w[1])) for _ in range(self.graph.number_of_nodes())]

    def __str__(self) -> str:
        return '\n'.join([f'{o}' for o in self.operations])
    
    def print(self):
        options = {
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 18,
        }

        pos = nx.planar_layout(self.graph, center = [10, 10], scale = 20)
        nx.draw(self.graph, pos, with_labels=True, font_color='white', **options)
        nx.draw_networkx_edge_labels(self.graph, pos, {(x, y): z['weight'] for (x, y, z) in nx.to_edgelist(self.graph)},font_color='red')

class NetworkOptimizationProblem(OptimizationProblem):
    def __init__(self, network_graph, task_graph, f_objective, f_constraints=None, bounds=None, dtype = int, t_lim = 20, net_speed = 100, name="NETproblem 1"):
        """
        :param network_graph: Граф сети, представленный с помощью NetGraph.
        :param task_graph: Граф задач, представленный с помощью TaskGraph.
        :param f_objective: Список целевых функций, которые мы хотим оптимизировать.
        :param f_constraints: Список функций-ограничений, которым должно соответствовать распределение.
        :param v_constraints: Список ограничений, которым должен соответствовать вектор (например, размер сети).
        """
        len_v = task_graph.graph.number_of_nodes()
        if bounds is None:
            constraints = [(0, network_graph.graph.number_of_nodes() - 1) for _ in range(len_v)]
        else:
            constraints = [(0, network_graph.graph.number_of_nodes() - 1) for _ in range(len_v)]
            for node, constr in bounds.items():
                constraints[node] = constr
        super().__init__(f_objective, f_constraints, bounds=constraints, dtype=dtype, len=len_v, name=name)
        self.network_graph = network_graph
        self.task_graph = task_graph
        self.t_lim = t_lim
        self.net_speed = net_speed