import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import math
import pandas as pd
import copy
from dataclasses import dataclass
from functools import reduce
import time
import seaborn as sns

@dataclass
class Node:
    p: int = 0
    e0: int = 1
    eMax: int = 1

class Net:
    def __init__(self, martx, nodes) -> None:
        self.nodes = nodes # machines
        self.graph = nx.Graph(np.array(martx))
        self.number = len(self.nodes)

    def __str__(self):
        return f'''nodes : {self.nodes}\n'''

    def print(self):
        pos = nx.spring_layout(self.graph, seed=100)
        nx.draw(self.graph, pos, with_labels=True, font_color='white')


@dataclass
class Operation:
    w: int = 0

class Task:
    def __init__(self, martx, operations) -> None:
        self.operations = operations
        self.graph = nx.DiGraph(np.array(martx))
        self.number = len(self.operations)

    def __str__(self):
        return f'''operations : {self.operations}\n'''

    def print(self):
        options = {
            'width': 1,
            'arrowstyle': '-|>',
            'arrowsize': 18,
        }

        pos = nx.planar_layout(self.graph, center = [10, 10], scale = 20)
        nx.draw(self.graph, pos, with_labels=True, font_color='white', **options)
        nx.draw_networkx_edge_labels(self.graph, pos, {(x, y): z['weight'] for (x, y, z) in nx.to_edgelist(self.graph)},font_color='red')


class Solution:
    _NET = None
    _TASK = None
    _CRITERIA = None
    _LIMITATIONS = None

    def __str__(self):
        return f'''Cвертка : {getattr(self, 'Свертка')},\nРаспределение: {self.distribution}, \nНагруженность на узлы: {self.W}, \nПроизводительность узлов {self._NET.nodes},\nНагруженоость на узлы {getattr(self, 'D')} \n{self.__paramets_to_str()}'''

    def __paramets_to_str(self):
        res = 'ЦФ\n'
        for k, v in self._CRITERIA['all/group'].items():
            res += k + ' / ' + v[1] + ':' + str(getattr(self, k)) + '\n'
        for k, v in self._CRITERIA['mashin'].items():
           for index_m in v['index_node']:
                res += k + ' m' + str(index_m) + ' / ' + v['f'][1] + ':' + str(getattr(self, k + ' m' + str(index_m))) + '\n'
        res += 'Ограничения\n'
        res += str(getattr(self, 'Выполнение условий')) + '\n'
        for k, v in self._LIMITATIONS['all/group'].items():
            res += k + ' / ' + ':' + str(getattr(self, k)) + '\n'
        for k, v in self._LIMITATIONS['mashin'].items():
           for index_m in v['index_node']:
                res += k + ' m' + str(index_m) + ' / ' + ':' + str(getattr(self, k + ' m' + str(index_m))) + '\n'
        return res

    def __eq__(self, other):
        return np.array_equal(self.distribution, other.distribution)

    def __init__(self, distribution = None):
        if distribution is None:
            self.set_random_distribution()
        else:
            self.set_distribution(distribution)
        self.velocity = np.random.uniform(-1, 1, self._TASK.number)
        self.best_position = copy.deepcopy(self)
        self.best_score = float('inf')

    def update_distribution(self):
        disrt = np.clip(np.round(self.distribution + self.velocity), 0, Solution._NET.number-1).astype(int)
        self.set_distribution(disrt)

    def update_velocity(self, global_best_position, omega, phi_p, phi_g):
        rp, rg = np.random.random(), np.random.random()
        self.velocity = (omega * self.velocity +
                         phi_p * rp * (self.best_position.distribution - self.distribution) +
                         phi_g * rg * (global_best_position.distribution - self.distribution))

    def set_random_distribution(self, CONST_DISTRIBUTION={}):
        self.set_distribution(np.random.randint(0, self._NET.number, size=self._TASK.number))

    def set_distribution(self, distr):
        self.distribution = np.array(distr)
        self.create_paths()
        self.network_status_calculation()
        self.calculation_limitations()
        self.calculation_objective_function()

    def update(self):
        self.create_paths()
        self.network_status_calculation()
        self.calculation_limitations()
        self.calculation_objective_function()

    def mutation(self, indpb=0.01):
        dist = self.distribution.copy()
        for task in range(self._TASK.number):
            if random.random() <= indpb:
                dist[task] = random.randint(0, self._NET.number-1)
        self.set_distribution(dist)

    def create_paths(self, indpb=0.01):
        # 'Операция1Операция2:ПутьВГрафеСети'
        self.paths = {}
        for start, end, _ in nx.to_edgelist(Solution._TASK.graph):
            if self.distribution[start] == self.distribution[end]:
                self.paths[str(start) + str(end)] = [self.distribution[start]]
            else:
                self.paths[str(start) + str(end)] = random.choice([item for item in nx.all_shortest_paths(Solution._NET.graph, self.distribution[start], self.distribution[end])])

    def network_status_calculation(self):
        # Подсчет трудоемкости
        self.W = [0 for _ in range(Solution._NET.number)]

        # Сколько каждый узел должен обработать задач
        self.v_task_to_node = [0 for _ in range(Solution._NET.number)]
        # Сколько каждый узел должен отправить
        self.v_sent_to_node = [0 for _ in range(Solution._NET.number)]
        # Сколько каждый узел должен принять
        self.v_reseive_to_node = [0 for _ in range(Solution._NET.number)]

        for start, end, weight in nx.to_edgelist(Solution._TASK.graph):
            """
            start - задача начало
            end - задача конец
            weight - то сколько должено быть прередано из одной задачи в другую
            """
            if not len(self.paths[str(start) + str(end)]) == 1:
                temp = self.paths[str(start) + str(end)].copy()
                # task_weight - нагрузка на выполнение задачи
                #Оброботка нагрузки начала пути

                self.W[temp[0]] += Solution._TASK.operations[start].w + weight['weight']

                self.v_sent_to_node[temp[0]] += weight['weight']
                self.v_task_to_node[temp[0]] += Solution._TASK.operations[start].w
                #Оброботка нагрузки конца пути

                self.W[temp[-1]] += Solution._TASK.operations[end].w + weight['weight']

                self.v_reseive_to_node[temp[-1]] += weight['weight']
                self.v_task_to_node[temp[-1]] += Solution._TASK.operations[start].w

                temp.pop(0)
                temp.pop(-1)
                # Оброботка нагрузки всех остальных узлов в пути
                for i in temp:
                    self.W[i] += 2 * weight['weight']
                    self.v_sent_to_node[i] += weight['weight']
                    self.v_reseive_to_node[i] += weight['weight']
            else:
                self.W[self.distribution[start]] += Solution._TASK.operations[start].w
                self.v_task_to_node[self.distribution[start]] += Solution._TASK.operations[start].w

                self.W[self.distribution[end]] += Solution._TASK.operations[end].w
                self.v_task_to_node[self.distribution[end]] += Solution._TASK.operations[end].w

    def calculation_objective_function(self):
        # Считаем все критрии вида all/group для объекта
        convolution = 1
        for k, v in self._CRITERIA['all/group'].items():
            setattr(self, k, v[0](self, v[2]))

            if v[1] == "max":
                convolution *= v[0](self, v[2])
            else:
                convolution *= 1/v[0](self, v[2]) if v[0](self, v[2])>0 else 0

        # Считаем все критрии для объекта
        for k, v in self._CRITERIA['mashin'].items():
            for index_m in v['index_node']:
                setattr(self, k + ' m' + str(index_m), v['f'][0](self, index_m, v['f'][2]))
                if v['f'][1] == "max":
                    convolution *=   v['f'][0](self, index_m, v['f'][2])
                else:
                    convolution *= 1/v['f'][0](self, index_m, v['f'][2]) if v['f'][0](self, index_m, v['f'][2])>0 else 0

        setattr(self, 'Свертка', convolution if getattr(self, 'Выполнение условий') > 0 else -1)

    def calculation_limitations(self):
        # Считаем все критрии вида all/group для объекта
        convolution = 1
        for k, v in self._LIMITATIONS['all/group'].items():
            value = v[0](self, v[1])
            setattr(self, k, value)
            convolution *= value

        # Считаем все критрии для объекта
        for k, v in self._LIMITATIONS['mashin'].items():
            for index_m in v['index_node']:
                value = v['f'][0](self, index_m, v['f'][1])
                setattr(self, k + ' m' + str(index_m), value)
                convolution *=  value

        setattr(self, 'Выполнение условий', convolution)