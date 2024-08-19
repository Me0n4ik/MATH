# opti/problem.py

import numpy as np
import networkx as nx
from dataclasses import dataclass
import functools

class OptimizationProblem:
    def __init__(self, f_objective, f_constraints=None, bounds=None, dtype = int, len = 10):
        """
        :param initial_vector: Исходный вектор, который мы оптимизируем.
        :param f_objective: Список целевых функций, которые мы хотим оптимизировать.
        :param f_constraints: Список функций-ограничений, которым должен соответствовать вектор.
        :param v_constraints: Список ограничений, которым должен соответствовать вектор.
        """
        self.f_objective = f_objective
        self.f_constraints = f_constraints if f_constraints is not None else []
        self.vector_length = None
        self.bounds = np.array(bounds) if bounds is not None else None
        self.dtype = dtype
        self.vector_length = len

    def generate_random_solution(self):
        if self.bounds is not None:
            lower_bounds, upper_bounds = self.bounds[:, 0], self.bounds[:, 1]
            return np.random.uniform(lower_bounds, upper_bounds, self.vector_length).astype(self.dtype)
        else:
            return np.zeros(self.vector_length, dtype=self.dtype)




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
        return - np.inf
    
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
    def __init__(self, martx, net_power = (1500, 2500), e0 = (0,70), emax = (70,100)) -> None:
        self.nodes = [NetNode(p, e1, e2) for p, e1, e2 in zip(  np.random.randint(net_power[0], net_power[1]),\
                                                                np.random.randint(e0[0], e0[1]), \
                                                                np.random.randint(emax[0], emax[1]))]
        self.graph = nx.Graph(np.array(martx))

@dataclass
class TaskNode:
    w: int = 0

class TaskGraph:
    def __init__(self, martx, w = (100,600)) -> None:
        self.graph = nx.DiGraph(np.array(martx))
        self.operations = [TaskNode(np.random.randint(w[0], w[1])) for _ in self.graph.number_of_nodes()]

class NetworkOptimizationProblem(OptimizationProblem):
    def __init__(self, network_graph, task_graph, f_objective, f_constraints=None, bounds=None, dtype = int):
        """
        :param network_graph: Граф сети, представленный с помощью NetGraph.
        :param task_graph: Граф задач, представленный с помощью TaskGraph.
        :param f_objective: Список целевых функций, которые мы хотим оптимизировать.
        :param f_constraints: Список функций-ограничений, которым должно соответствовать распределение.
        :param v_constraints: Список ограничений, которым должен соответствовать вектор (например, размер сети).
        """
        len_v =task_graph.number_of_nodes()
        if bounds is None:
            constraints = [(0, network_graph.number_of_nodes() - 1) for _ in range(task_graph.number_of_nodes())],
        else:
            constraints = [(0, network_graph.number_of_nodes() - 1) for _ in range(task_graph.number_of_nodes())]
            for node, constr in bounds.items():
                constraints[node] = constr

        super().__init__(f_objective, f_constraints, bounds=constraints, dtype=dtype, len=len_v)
        self.network_graph = network_graph
        self.task_graph = task_graph

    def network_status_calculation(self, vector):
        # Подсчет трудоемкости
        W = [0 for _ in range(self.network_graph.graph.number_of_nodes())]
        # Сколько каждый узел должен обработать задач
        v_task_to_node = [0 for _ in range(self.network_graph.graph.number_of_nodes())]
        # Сколько каждый узел должен отправить
        v_sent_to_node = [0 for _ in range(self.network_graph.graph.number_of_nodes())]
        # Сколько каждый узел должен принять
        v_reseive_to_node = [0 for _ in range(self.network_graph.graph.number_of_nodes())]

        for start, end, weight in nx.to_edgelist(self.task_graph.graph):
            """
            start - задача начало
            end - задача конец
            weight - то сколько должено быть прередано из одной задачи в другую
            """
            if not len(paths[str(start) + str(end)]) == 1:
                temp = paths[str(start) + str(end)].copy()
                # task_weight - нагрузка на выполнение задачи
                #Оброботка нагрузки начала пути

                W[temp[0]] += self.task_graph.operations[start].w + weight['weight']

                v_sent_to_node[temp[0]] += weight['weight']
                v_task_to_node[temp[0]] += self.task_graph.operations[start].w
                #Оброботка нагрузки конца пути

                self.W[temp[-1]] += self.task_graph.operations[end].w + weight['weight']

                self.v_reseive_to_node[temp[-1]] += weight['weight']
                self.v_task_to_node[temp[-1]] += self.task_graph.operations[start].w

                temp.pop(0)
                temp.pop(-1)
                # Оброботка нагрузки всех остальных узлов в пути
                for i in temp:
                    W[i] += 2 * weight['weight']
                    v_sent_to_node[i] += weight['weight']
                    v_reseive_to_node[i] += weight['weight']
            else:
                W[self.distribution[start]] += self.task_graph.operations[start].w
                v_task_to_node[self.distribution[start]] += self.task_graph.operations[start].w

                W[self.distribution[end]] += self.task_graph.operations[end].w
                v_task_to_node[self.distribution[end]] += self.task_graph.operations[end].w