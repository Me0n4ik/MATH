import numpy as np
import logging
import networkx as nx
from dataclasses import dataclass

from .main import Vector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Класс для описания задачи оптимизации
class OptimizationProblem:
    def __init__(self, f_objective = [], f_constraints = [], v_constraints = {}):
        """
        :param initial_vector: Исходный вектор, который мы оптимизируем.
        :param f_objective: Список целевых функций, которые мы хотим оптимизировать.
        :param f_constraints: Список функций-ограничений, которым должен соответствовать вектор.
        :param v_constraints: Список ограничений, которым должен соответствовать вектор.
        """
        self.f_objective = f_objective
        self.f_constraints = f_constraints
        self.set_vector_const(v_constraints)

    def set_vector_const(self, v_constraints = {}):
        for key, item in v_constraints.items():
            match key:
                case 'len':
                    print(v_constraints)
                    Vector.length_constraint = item
                case 'constraints':
                    Vector.element_constraints = np.array(item)
                case 'dtype':
                    Vector.dtype = item
                    
    def evaluate_objectives(self, vector=None):
        """Оценивает все целевые функции на заданном векторе. Возвращает список значений.""" 
        return [f(vector, self) for f in self.f_objective]
    
    def convolution_evaluate_objectives(self, vector=None):
        return np.prod([f(vector, self) for f in self.f_objective])

    def expanded_constraints(self, vector=None):
        return [c(vector, self) for c in self.f_constraints]

    def check_constraints(self, vector=None):
        """Проверяет, выполняются ли все ограничения на заданном векторе."""
        return all(c(vector, self) for c in self.f_constraints)

    def print_info(self):
        """Выводит информацию о целевых функциях и ограничениях."""
        print("Целевые функции:")
        for f in self.f_objective:
            print(f" - {f.__name__}")

        print("Ограничения:")
        for c in self.f_constraints:
            print(f" - {c.__name__}")


@dataclass
class NetNode:
    p: int = 0
    e0: int = 1
    eMax: int = 1

class NetGraph:
    def __init__(self, martx, net_power = (), e0 = (), emax = ()) -> None:
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
    def __init__(self, network_graph, task_graph, f_objective=[], f_constraints=[], v_constraint={}):
        """
        :param network_graph: Граф сети, представленный с помощью NetGraph.
        :param task_graph: Граф задач, представленный с помощью TaskGraph.
        :param f_objective: Список целевых функций, которые мы хотим оптимизировать.
        :param f_constraints: Список функций-ограничений, которым должно соответствовать распределение.
        :param v_constraints: Список ограничений, которым должен соответствовать вектор (например, размер сети).
        """
        if len(v_constraint) == 0:
            v_constraints = {
                'len': task_graph.number_of_nodes(),
                'constraints': [(0, network_graph.number_of_nodes() - 1) for _ in range(task_graph.number_of_nodes())],
                'dtype': int
            }
        else:
            constraints = [(0, network_graph.number_of_nodes() - 1) for _ in range(task_graph.number_of_nodes())]
            for node, constr in v_constraint.items():
                constraints[node] = constr
            
            v_constraints = {
                'len': task_graph.number_of_nodes(),
                'constraints': constraints,
                'dtype': int
            }

        super().__init__(f_objective, f_constraints, v_constraints)
        self.network_graph = network_graph
        self.task_graph = task_graph

    def network_status_calculation(self):
        # Подсчет трудоемкости
        self.W = [0 for _ in range(self.problem.network_graph.graph.number_of_nodes())]
        # Сколько каждый узел должен обработать задач
        self.v_task_to_node = [0 for _ in range(self.problem.network_graph.graph.number_of_nodes())]
        # Сколько каждый узел должен отправить
        self.v_sent_to_node = [0 for _ in range(self.problem.network_graph.graph.number_of_nodes())]
        # Сколько каждый узел должен принять
        self.v_reseive_to_node = [0 for _ in range(self.problem.network_graph.graph.number_of_nodes())]

        for start, end, weight in nx.to_edgelist(self.problem.task_graph.graph):
            """
            start - задача начало
            end - задача конец
            weight - то сколько должено быть прередано из одной задачи в другую
            """
            if not len(self.paths[str(start) + str(end)]) == 1:
                temp = self.paths[str(start) + str(end)].copy()
                # task_weight - нагрузка на выполнение задачи
                #Оброботка нагрузки начала пути

                self.W[temp[0]] += self.problem.task_graph.operations[start].w + weight['weight']

                self.v_sent_to_node[temp[0]] += weight['weight']
                self.v_task_to_node[temp[0]] += self.problem.task_graph.operations[start].w
                #Оброботка нагрузки конца пути

                self.W[temp[-1]] += self.problem.task_graph.operations[end].w + weight['weight']

                self.v_reseive_to_node[temp[-1]] += weight['weight']
                self.v_task_to_node[temp[-1]] += self.problem.task_graph.operations[start].w

                temp.pop(0)
                temp.pop(-1)
                # Оброботка нагрузки всех остальных узлов в пути
                for i in temp:
                    self.W[i] += 2 * weight['weight']
                    self.v_sent_to_node[i] += weight['weight']
                    self.v_reseive_to_node[i] += weight['weight']
            else:
                self.W[self.distribution[start]] += self.problem.task_graph.operations[start].w
                self.v_task_to_node[self.distribution[start]] += self.problem.task_graph.operations[start].w

                self.W[self.distribution[end]] += self.problem.task_graph.operations[end].w
                self.v_task_to_node[self.distribution[end]] += self.problem.task_graph.operations[end].w