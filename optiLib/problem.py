# opti/problem.py

import numpy as np
import networkx as nx
from dataclasses import dataclass
import functools
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from typing import Tuple, Dict, List
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np



class OptimizationProblem:
    """
    Класс для решения задач оптимизации с множеством целевых функций и ограничений.

    Класс предоставляет функционал для:
    - Определения и оценки множества целевых функций
    - Задания ограничений на решения
    - Работы со специальными функциями для конкретных узлов
    - Генерации случайных допустимых решений
    - Проверки допустимости решений
    - Вычисления свертки целевых функций

    Атрибуты:
    ----------
    name : str
        Название задачи оптимизации
    
    f_objective : list
        Список основных целевых функций
    
    constraints : list
        Список функций-ограничений
    
    vector_length : int
        Длина вектора решения
    
    bounds : numpy.ndarray
        Границы допустимых значений для переменных
    
    dtype : type
        Тип данных элементов вектора решения
    
    node_functions : dict
        Словарь специальных функций для конкретных узлов

    function_constraints : dict
        Словарь ограничений для функций {функция: [мин_значение, макс_значение]}
    
    special_function_constraints : dict
        Словарь ограничений для специальных функций узлов 
        {(узлы): {функция: [мин_значение, макс_значение]}}   

    Методы:
    -------
    generate_random_solution()
        Генерирует случайное допустимое решение
    
    get_info(vector)
        Возвращает словарь со значениями всех функций для решения
    
    evaluate_objectives(vector)
        Вычисляет значения всех целевых функций
    
    convolution_evaluate_objectives(vector)
        Вычисляет свертку целевых функций
    
    expanded_constraints(vector)
        Проверяет выполнение всех ограничений
    
    check_constraints(vector)
        Проверяет допустимость решения
    
    evaluate(solution)
        Вычисляет итоговое значение для решения
    
    constrain_elements(vector)
        Приводит решение к допустимому виду
    
    is_feasible(solution)
        Проверяет допустимость решения
    

    Примечания:
    -----------
    1. Все целевые функции и ограничения должны принимать два аргумента:
       - vector: текущий вектор решения
       - problem: экземпляр задачи оптимизации
    
    2. Специальные функции для узлов применяются только когда 
       соответствующие узлы присутствуют в решении
    
    3. При создании экземпляра класса можно указать:
       - Множество целевых функций
       - Множество ограничений
       - Границы допустимых значений
       - Специальные функции для конкретных узлов
    
    4. Класс поддерживает как целочисленную, так и вещественную оптимизацию
       через параметр dtype

    4. Все функции минимизируются
    """

    def __init__(self, f_objective, constraints=None, bounds=None, dtype = int, len = 10, name = "Problem 1", node_functions=None, 
                 function_constraints=None, special_function_constraints=None):
        """
        Инициализация задачи оптимизации 
        
        Параметры:
        -----------
        f_objective : list
            Список основных целевых функций для оптимизации
        
        constraints : list, optional
            Список функций-ограничений, которым должно удовлетворять решение
            
        bounds : array-like, optional 
            Границы допустимых значений для переменных в формате [[min1,max1], [min2,max2],...]
            
        dtype : type, optional
            Тип данных для элементов вектора решения (по умолчанию int)
            
        len : int, optional
            Длина вектора решения (по умолчанию 10)
            
        name : str, optional
            Название задачи оптимизации (по умолчанию "Problem 1")
            
        node_functions : dict, optional
            Словарь специальных функций для конкретных узлов
            Формат: {(узел1, узел2,...): функция, ...}
            Пример: {(1,2): special_func1, (3,4,5): special_func2}
                
        function_constraints : dict
            Словарь ограничений для функций {функция: [мин_значение, макс_значение]}
            Пример:     function_constraints={
                            objective1: [0, 100]  # ограничение на основную функцию
                        }

        special_function_constraints : dict
            Словарь ограничений для специальных функций узлов 
            {(узлы): {функция: [мин_значение, макс_значение]}}
            Пример :    special_function_constraints={
                            (1, 2): {special_func1: [0, 10]}  # ограничение на специальную функцию
                        }
        """
        self.name = name
        self.f_objective = f_objective
        self.constraints = constraints if constraints is not None else []
        self.vector_length = None
        self.bounds = np.array(bounds) if bounds is not None else None
        self.dtype = dtype
        self.vector_length = len
        self.node_functions = node_functions if node_functions is not None else {}
        self.function_constraints = function_constraints if function_constraints is not None else {}
        self.special_function_constraints = special_function_constraints if special_function_constraints is not None else {}

    def generate_random_solution(self):
        """
        Генерирует случайное допустимое решение задачи.
        
        Алгоритм:
        1. С вероятностью 0.7 генерирует решение на основе выбора узлов сети
        2. С вероятностью 0.3 генерирует решение в заданных границах
        
        Возвращает:
        -----------
        numpy.ndarray
            Случайный допустимый вектор решения
        """
        if np.random.random() < 0.7:
            net_nodes = self.network_graph.graph.number_of_nodes()
            tasks = self.task_graph.graph.number_of_nodes()
            num_to_select = np.random.randint(1, net_nodes + 1)
            nods_to_select = np.random.choice([i for i in range(net_nodes)], size=num_to_select, replace=False)
            return self.constrain_elements(np.random.choice(nods_to_select, size=tasks))
        else:
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds[:, 0], self.bounds[:, 1]
                return self.constrain_elements(np.random.uniform(lower_bounds, upper_bounds, self.vector_length).astype(self.dtype))
            else:
                return self.constrain_elements(np.zeros(self.vector_length, dtype=self.dtype))

    def get_info_save(self, vector=None):
        """
        Возвращает расширенную информацию о решении.
        
        Параметры:
        ----------
        vector : array-like, optional
            Вектор решения для анализа
            
        Возвращает:
        -----------
        dict
            Словарь со значениями всех функций, их ограничениями и сверткой
        """
        info = {
            **{f.__name__: f(vector, self) for name, f in self.f_objective.items()},
            **{f.__name__: f(vector, self) for f in self.constraints},
            'Свертка': self.evaluate(vector)
        }
        # Добавляем информацию об ограничениях функций
        info['Ограничения функций'] = {
            f.__name__: {
                'значение': f(vector, self),
                'ограничения': bounds
            } for f, bounds in self.function_constraints.items()
        }

        # Добавляем информацию об ограничениях специальных функций
        info['Ограничения специальных функций'] = {}

        # Проходим по всем узлам и их ограничениям
        for nodes, constraints in self.special_function_constraints.items():
            # Проверяем, есть ли хотя бы один узел из nodes в vector
            has_node = False
            for node in nodes:
                if node in vector:
                    has_node = True
                    break
                    
            if has_node:
                # Создаем ключ для текущих узлов
                node_key = f'узлы {nodes}'
                info['Ограничения специальных функций'][node_key] = {}
                for node in nodes:
                    # Проходим по всем функциям и их ограничениям
                    for f, bounds in constraints.items():
                        # Добавляем информацию о функции
                        info['Ограничения специальных функций'][node_key][f.__name__] = {
                            'значение': f(vector, self, node),
                            'ограничения': bounds,
                            'узел': nodes  # Добавляем информацию об узле
                        }

        return info
    
    def get_info(self, vector=None):
        """
        Возвращает расширенную информацию о решении в форматированном виде.
        
        Параметры:
        ----------
        vector : array-like, optional
            Вектор решения для анализа
            
        Возвращает:
        -----------
        str
            Форматированный текст с информацией о решении
        """
        if vector is None:
            return "Вектор решения не предоставлен"

        info_str = "\n=== ИНФОРМАЦИЯ О РЕШЕНИИ ===\n\n"

        # Основные целевые функции
        info_str += "📊 ЦЕЛЕВЫЕ ФУНКЦИИ:\n"
        info_str += "-" * 40 + "\n"
        for name, f in self.f_objective.items():
            value = f(vector, self)
            info_str += f"▪ {f.__name__:<30} = {value if value is not None else 0:.4f}\n"
        info_str += "\n"

        # Ограничения
        info_str += "🔒 ОГРАНИЧЕНИЯ:\n"
        info_str += "-" * 40 + "\n"
        for f in self.constraints:
            value = f(vector, self)
            info_str += f"▪ {f.__name__:<30} = {value:.4f}\n"
        info_str += "\n"

        # Значение свертки
        info_str += "📈 ЗНАЧЕНИЕ СВЕРТКИ:\n"
        info_str += "-" * 40 + "\n"
        info_str += f"▪ Общая свертка = {self.evaluate(vector):.4f}\n\n"

        # Ограничения функций
        info_str += "🎯 ОГРАНИЧЕНИЯ ФУНКЦИЙ:\n"
        info_str += "-" * 40 + "\n"
        for f, bounds in self.function_constraints.items():
            value = f(vector, self)
            info_str += f"▪ {f.__name__}:\n"
            info_str += f"  ├─ Значение: {value:.4f}\n"
            info_str += f"  └─ Границы: [{bounds[0]}, {bounds[1]}]\n"
        info_str += "\n"

        # Специальные ограничения
        info_str += "⭐ СПЕЦИАЛЬНЫЕ ОГРАНИЧЕНИЯ:\n"
        info_str += "-" * 40 + "\n"
        for nodes, constraints in self.special_function_constraints.items():
            has_node = any(node in vector for node in nodes)
            
            if has_node:
                info_str += f"▪ Узлы {nodes}:\n"
                for f, bounds in constraints.items():
                    for node in nodes:
                        value = f(vector, self, node)
                        info_str += f"  ├─ {f.__name__} (узел {node}):\n"
                        info_str += f"  │  ├─ Значение: {value if value is not None else 0:.4f}\n"
                        info_str += f"  │  └─ Границы: [{bounds[0]}, {bounds[1]}]\n"
                info_str += "  └─\n"

        info_str += "\n=== КОНЕЦ ОТЧЕТА ===\n"

        return info_str

    def evaluate_objectives(self, vector=None):
        """
        Вычисление значений всех целевых функций для заданного вектора решения
        
        Параметры:
        -----------
        vector : array-like, optional
            Вектор решения для оценки. Если None, используется нулевой вектор
            
        Возвращает:
        -----------
        list
            Список значений всех целевых функций:
            - Сначала идут значения основных целевых функций
            - Затем значения специальных функций для конкретных узлов
            
        Примечания:
        -----------
        1. Для каждой основной целевой функции вычисляется значение
        2. Для каждой специальной функции проверяется наличие соответствующих узлов 
           в векторе решения и, если они есть, вычисляется значение функции
        """
        # Вычисляем значения основных целевых функций
        base_objectives = [f(vector, self) for f in self.f_objective]
        
        # Вычисляем значения специальных функций для конкретных узлов
        node_specific_objectives = []
        if vector is not None:
            for nodes, func in self.node_functions.items():
                # Проверяем наличие узлов из массива в текущем векторе
                for node in nodes:
                    if node in vector:
                        node_specific_objectives.append(func(vector, self))

        # Возвращаем все значения целевых функций
        return base_objectives + node_specific_objectives
    
    def convolution_evaluate_objectives(self, vector=None):
        """
        Вычисление свертки всех целевых функций
        
        Параметры:
        -----------
        vector : array-like, optional
            Вектор решения для оценки
            
        Возвращает:
        -----------
        float
            Произведение значений всех целевых функций (включая специальные)
            
        Примечания:
        -----------
        Свертка выполняется путем перемножения:
        - Значений основных целевых функций
        - Значений специальных функций для узлов
        """
        all_objectives = self.evaluate_objectives(vector)
        return np.prod(np.array([x for x in all_objectives if x is not None]))

    def check_constraints(self, vector=None):
        """
        Проверяет выполнение всех ограничений на ветор для заданного вектора.
        
        Параметры:
        ----------
        vector : array-like, optional
            Вектор решения для проверки
            
        Возвращает:
        -----------
        bool
            True если все ограничения выполнены, False иначе
        """
        return all(c(vector, self) for c in self.constraints)

    def expanded_constraints(self, vector=None):
        """
        Проверяет выполнение всех функций-ограничений на задачу для заданного вектора.
        
        Параметры:
        ----------
        vector : array-like, optional
            Вектор решения для проверки
            
        Возвращает:
        -----------
        list
            Список булевых значений для каждого ограничения
            
        Примечания:
        -----------
        True означает, что ограничение выполняется
        False означает нарушение ограничения
        """
        return [c(vector, self) for c in self.constraints]

    def check_function_constraints(self, vector):
        """
        Проверяет выполнение ограничений на значения функций.
        
        Параметры:
        ----------
        vector : array-like
            Вектор решения для проверки
            
        Возвращает:
        -----------
        bool
            True если все ограничения на функции выполнены, False иначе
            
        Примечания:
        -----------
        Проверяет ограничения как для основных, так и для специальных функций
        """
        # Проверка ограничений основных функций
        for func, bounds in self.function_constraints.items():
            value = func(vector, self)
            if not (bounds[0] <= value <= bounds[1]):
                return False
                
        # Проверка ограничений специальных функций
        for nodes, constraints in self.special_function_constraints.items():
                for node in nodes: 
                    if node in vector:
                        for func, bounds in constraints.items():
                            value = func(vector, self, node)
                            if value is not None:
                                if not (bounds[0] <= value <= bounds[1]):
                                    return False
                        
        return True
    
    def constrain_elements(self, vector):
        """
        Приводит элементы вектора к допустимым значениям согласно ограничениям.
        
        Параметры:
        ----------
        vector : array-like
            Исходный вектор решения
            
        Возвращает:
        -----------
        numpy.ndarray
            Вектор с элементами, приведенными к допустимым значениям
            
        Примечания:
        -----------
        1. Преобразует вектор к заданному типу данных
        2. Если заданы границы, обрезает значения по этим границам
        """
        vector = np.array(vector).astype(self.dtype)

        if self.bounds is not None:
            lower_bounds, upper_bounds = self.bounds[:, 0], self.bounds[:, 1]
            return np.clip(vector, lower_bounds, upper_bounds)
        return vector

    def is_feasible(self, solution):
        """
        Проверяет допустимость решения.
        
        Параметры:
        ----------
        solution : array-like
            Вектор решения для проверки
            
        Возвращает:
        -----------
        bool
            True если решение допустимо, False иначе
            
        Примечания:
        -----------
        Решение считается допустимым если:
        1. Длина вектора соответствует требуемой
        2. Выполнены все ограничения
        """
        if len(solution) != self.vector_length:
            return False
        return self.check_constraints(solution) and self.check_function_constraints(solution)

    def evaluate(self, solution):
        """
        Вычисляет итоговое значение целевой функции для решения.
        
        Параметры:
        ----------
        solution : array-like
            Вектор решения для оценки
            
        Возвращает:
        -----------
        float
            Значение свертки целевых функций если решение допустимо,
            бесконечность если решение недопустимо
            
        Примечания:
        -----------
        1. Сначала решение приводится к допустимому виду
        2. Проверяется выполнение всех ограничений
        3. Вычисляется свертка целевых функций
        """
        solution = self.constrain_elements(solution)
        if self.is_feasible(solution):
            return self.convolution_evaluate_objectives(solution)
        return np.inf

class IntegerOptimizationProblem(OptimizationProblem):
    def __init__(self, f_objective, constraints=None, bounds=None, len = 10):
        super().__init__(f_objective, constraints=None, bounds=None, dtype = int, len = 10)

    def evaluate(self, solution):
        return super().evaluate(solution)


@dataclass
class NetworkNode:
    id: int
    performance: float  # Производительность
    e_receive: float = 0.0  # Энергозатраты на прием
    e_comp: float    = 0.0  # Энергозатраты на вычисления
    e_send: float    = 0.0  # Энергозатраты на отправку
    cost: float      = 0.0  # Стоимость устройства
    failure_rate: float = 0.0  # Интенсивность отказов

    def __str__(self) -> str:
        return f"Node {self.id} - Performance: {self.performance}"

    def get_node_performance(self) -> float:
        """
        Возвращает производительность узла
        """
        # Этот метод должен быть реализован в зависимости от вашей структуры сети
        return self.performance


class NetGraph:
    def __init__(self, graph_type: int = 1, matrix=None, net_power=(100, 2500)) -> None:
        """
        Инициализация графа сети

        Args:
            graph_type: Тип графа (1 - сложный, 2 - линейный)
            matrix: Матрица смежности (если None, используется предустановленная топология)
            net_power: Диапазон производительности для случайной генерации
        """
        self.graph_type = graph_type
        if matrix is not None:
            self.graph = nx.Graph(np.array(matrix))
            self._create_nodes_from_matrix(net_power)
        else:
            self.graph, self.nodes = self._create_predefined_network()

    def _create_nodes_from_matrix(self, net_power):
        """Создание узлов на основе матрицы смежности"""
        self.nodes = {}
        for i in range(self.graph.number_of_nodes()):
            self.nodes[i] = NetworkNode(
                id=i,
                performance=np.random.randint(net_power[0], net_power[1]),
                e_receive=np.random.uniform(0.1, 0.5),
                e_comp=np.random.uniform(0.2, 0.8),
                e_send=np.random.uniform(0.1, 0.5),
                cost=np.random.randint(net_power[0], net_power[1]) * np.random.uniform(0.8, 1.2)
            )

    def _create_predefined_network(self):
        """Создание предустановленной сети"""
        if self.graph_type == 1:
            return self._create_complex_network()
        else:
            return self._create_linear_network()

    def _create_complex_network(self):
        """Создание сложной сети (первый пример)"""
        network = nx.Graph()

        # Производительность узлов
        performances = {
            0: 100, 1: 500, 2: 500, 3: 1000,
            4: 1000, 5: 1000, 6: 5000, 7: 5000, 8: 5000
        }

        # Ребра с пропускной способностью
        edges = [
            (0, 2, 500), (0, 1, 500), (1, 4, 1000),
            (1, 3, 1000), (2, 4, 1000), (2, 5, 1000),
            (5, 6, 5000), (5, 7, 5000), (4, 7, 5000),
            (4, 8, 5000), (3, 8, 5000)
        ]

        nodes = self._create_nodes(performances)
        self._add_edges(network, edges, nodes)
        return network, nodes

    def _create_linear_network(self):
        """Создание линейной сети (второй пример)"""
        network = nx.Graph()

        # Производительность узлов
        performances = {
            0: 1000, 1: 1500, 2: 2000,
            3: 3000, 4: 10000
        }

        # Ребра с пропускной способностью
        edges = [
            (0, 1, 500), (1, 2, 500),
            (2, 3, 500), (3, 4, 500)
        ]

        nodes = self._create_nodes(performances)
        self._add_edges(network, edges, nodes)
        return network, nodes

    def _create_nodes(self, performances):
        """Создание узлов с заданными характеристиками"""
        nodes = {}
        for node_id, perf in performances.items():
            nodes[node_id] = NetworkNode(
                id=node_id,
                performance=perf,
                e_receive=np.random.uniform(0.1, 0.5),
                e_comp=np.random.uniform(0.2, 0.8),
                e_send=np.random.uniform(0.1, 0.5),
                cost=perf * np.random.uniform(0.8, 1.2)
            )
        return nodes

    def _add_edges(self, network, edges, nodes):
        """Добавление узлов и ребер в сеть"""
        for node in nodes.values():
            network.add_node(node.id)
        for (u, v, bandwidth) in edges:
            network.add_edge(u, v, bandwidth=bandwidth)

    def visualize(self):
        """Визуализация сети"""
        if self.graph_type == 1:
            self._visualize_complex()
        else:
            self._visualize_linear()

    def _visualize_complex(self):
        """Визуализация сложной сети"""
        plt.figure(figsize=(15, 10))

        pos = {
            0: (-2, 0), 1: (-1, -1), 2: (-1, 1),
            3: (0, -2), 4: (0, 0), 5: (0, 2),
            6: (1, 3), 7: (1, 1), 8: (1, -1)
        }

        self._draw_network(pos)

    def _visualize_linear(self):
        """Визуализация линейной сети"""
        plt.figure(figsize=(15, 5))

        pos = {i: (i, 0) for i in range(len(self.nodes))}

        self._draw_network(pos)

    def _draw_network(self, pos):
        """Отрисовка сети"""
        ax = plt.gca()

        # Рисуем узлы
        node_colors = [self.nodes[n].get_node_performance() for n in self.graph.nodes()]
        nodes_draw = nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=1500,
            cmap=plt.cm.viridis,
            ax=ax
        )

        # Рисуем ребра
        nx.draw_networkx_edges(
            self.graph, pos,
            width=2,
            edge_color='gray',
            alpha=0.6
        )

        # Добавляем метки
        edge_labels = nx.get_edge_attributes(self.graph, 'bandwidth')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=10)

        labels = {n: f"Node {n}\n({self.nodes[n].get_node_performance()})" for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)

        plt.colorbar(nodes_draw, ax=ax, label='Performance')

        plt.title(f"{'Complex' if self.graph_type == 1 else 'Linear'} Network Topology")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def print_info(self):
        """Вывод информации о сети"""
        print("\nNetwork Properties:")
        print(f"Network type: {'Complex' if self.graph_type == 1 else 'Linear'}")
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")
        print("\nNodes:")
        for node in self.nodes.values():
            print(str(node))
        print("\nEdge bandwidths:")
        for (u, v, data) in self.graph.edges(data=True):
            print(f"Edge {u}-{v}: {data['bandwidth']}")

    def get_node_by_id(self, node_id: int) -> NetworkNode:
        """
        Возвращает узел сети по его ID
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise ValueError(f"Узел с ID {node_id} не найден")    

@dataclass
class Task:
    id: int
    complexity: float  # Вычислительная сложность
    input_data: float  # Объем входных данных
    output_data: float # Объем выходных данных
    deadline: float    # Предельное время выполнения

    def get_task_complexity(self) -> float:
        """
        Возвращает производительность узла
        """
        # Этот метод должен быть реализован в зависимости от вашей структуры сети
        return self.complexity
    

class TaskGraph:
    def __init__(self, graph_type: int = 1):
        """
        Инициализация графа задач

        Args:
            graph_type: Тип графа (1 - сложный, 2 - простой, 3 - полносвязный)
        """
        self.graph_type = graph_type
        self.graph, self.operations = self._create_task_graph()

    def _create_task_graph(self) -> Tuple[nx.DiGraph, Dict[int, Task]]:
        """Создание графа задач выбранного типа"""
        if self.graph_type == 1:
            return self._create_complex_task_graph()
        elif self.graph_type == 2:
            return self._create_simple_task_graph()
        else:
            return self._create_fully_connected_task_graph()

    def _create_complex_task_graph(self) -> Tuple[nx.DiGraph, Dict[int, Task]]:
        """Создание сложного графа задач (первый пример)"""
        task_graph = nx.DiGraph()

        # Параметры задач
        tasks_params = {
            0: {"complexity": 100, "deadline": 500},  # Начальная задача
            1: {"complexity": 300, "deadline": 600},  # Верхняя ветвь
            2: {"complexity": 200, "deadline": 700},
            3: {"complexity": 100, "deadline": 800},
            4: {"complexity": 300, "deadline": 600},  # Средняя ветвь
            5: {"complexity": 200, "deadline": 700},
            6: {"complexity": 100, "deadline": 800},
            7: {"complexity": 300, "deadline": 600},  # Нижняя ветвь
            8: {"complexity": 200, "deadline": 700},
            9: {"complexity": 50, "deadline": 1000}   # Конечная задача
        }

        # Ребра с объемами передаваемых данных
        edges = [
            (0, 1, 10000), (1, 2, 2000), (2, 3, 1000), (3, 9, 500),
            (0, 4, 10000), (4, 5, 2000), (5, 6, 1000), (6, 9, 500),
            (0, 7, 10000), (7, 8, 2000), (8, 6, 500)
        ]

        return self._create_graph_structure(task_graph, tasks_params, edges)

    def _create_simple_task_graph(self) -> Tuple[nx.DiGraph, Dict[int, Task]]:
        """Создание простого графа задач (второй пример)"""
        task_graph = nx.DiGraph()

        tasks_params = {
            0: {"complexity": 1000, "deadline": 500},
            1: {"complexity": 5000, "deadline": 700},
            2: {"complexity": 5000, "deadline": 700},
            3: {"complexity": 5000, "deadline": 700},
            4: {"complexity": 1000, "deadline": 1000}
        }

        edges = [
            (0, 1, 1000), (0, 2, 1000), (0, 3, 1000),
            (1, 4, 1000), (2, 4, 1000), (3, 4, 1000)
        ]

        return self._create_graph_structure(task_graph, tasks_params, edges)

    def _create_fully_connected_task_graph(self) -> Tuple[nx.DiGraph, Dict[int, Task]]:
        """Создание полносвязного графа задач (третий пример)"""
        task_graph = nx.DiGraph()

        tasks_params = {
            0: {"complexity": 10000, "deadline": 500},
            1: {"complexity": 10000, "deadline": 500},
            2: {"complexity": 10000, "deadline": 500},
            3: {"complexity": 10000, "deadline": 500}
        }

        edges = [
            (0, 1, 500), (0, 2, 500), (0, 3, 500),
            (1, 2, 500), (1, 3, 500),
            (2, 1, 500), (2, 3, 500),
            (3, 1, 500), (3, 2, 500)
        ]

        return self._create_graph_structure(task_graph, tasks_params, edges)

    def _create_graph_structure(self, task_graph: nx.DiGraph,
                              tasks_params: Dict, edges: List) -> Tuple[nx.DiGraph, Dict[int, Task]]:
        """Создание структуры графа"""
        tasks = {}

        # Создаем узлы графа и задачи
        for task_id, params in tasks_params.items():
            task_graph.add_node(task_id)
            tasks[task_id] = Task(
                id=task_id,
                complexity=params["complexity"],
                input_data=0.0,  # Будет обновлено после добавления ребер
                output_data=0.0, # Будет обновлено после добавления ребер
                deadline=params["deadline"]
            )

        # Добавляем ребра и обновляем входные/выходные данные
        for (u, v, data) in edges:
            task_graph.add_edge(u, v, data_volume=data)
            tasks[v].input_data += data
            tasks[u].output_data += data

        return task_graph, tasks

    def visualize(self):
        """Визуализация графа задач"""
        plt.figure(figsize=(15, 10))
        ax = plt.gca()

        # Определяем позиции узлов в зависимости от типа графа
        if self.graph_type == 1:
            pos = {
                0: (-2, 0), 1: (0, 2), 2: (2, 2), 3: (4, 2),
                4: (0, 0), 5: (2, 0), 6: (4, 0),
                7: (0, -2), 8: (2, -2), 9: (6, 0)
            }
        elif self.graph_type == 2:
            pos = {
                0: (-2, 0),
                1: (0, 2), 2: (0, 0), 3: (0, -2),
                4: (2, 0)
            }
        else:
            pos = {
                0: (-1, 1), 1: (1, 1),
                2: (-1, -1), 3: (1, -1)
            }

        self._draw_graph(ax, pos)

    def _draw_graph(self, ax, pos):
        """Отрисовка графа"""
        # Рисуем узлы
        node_colors = [self.operations[n].complexity for n in self.graph.nodes()]
        nodes_draw = nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=1500,
            cmap=plt.cm.viridis,
            ax=ax
        )

        # Рисуем направленные ребра
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='gray',
            width=2,
            arrowsize=25,  # Увеличенный размер стрелок
            arrowstyle='-|>',  # Явно заданный стиль стрелок
            connectionstyle='arc3, rad=0.1',  # Изогнутые линии для лучшей видимости направления
            min_source_margin=25,  # Отступ от начала стрелки
          min_target_margin=25   # Отступ от конца стрелки
        )

        # Добавляем метки
        edge_labels = nx.get_edge_attributes(self.graph, 'data_volume')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=10)

        labels = {n: f"Task {n}\n({self.operations[n].complexity})"
                 for n in self.graph.nodes()}
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)

        # Добавляем colorbar
        plt.colorbar(nodes_draw, ax=ax, label='Computational Complexity')

        # Находим и выделяем критический путь
        critical_path = self.find_critical_path()
        if critical_path:
            path_edges = list(zip(critical_path[:-1], critical_path[1:]))
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=path_edges,
                edge_color='red',
                width=3
            )

        plt.title(f"Task Graph Type {self.graph_type} with Critical Path")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def find_critical_path(self) -> List[int]:
        """Находит критический путь в графе задач"""
        sources = [n for n in self.graph.nodes()
                  if self.graph.in_degree(n) == 0]
        sinks = [n for n in self.graph.nodes()
                if self.graph.out_degree(n) == 0]

        if not sources or not sinks:
            return None

        max_length = 0
        critical_path = None

        for source in sources:
            for sink in sinks:
                for path in nx.all_simple_paths(self.graph, source, sink):
                    length = self._calculate_path_length(path)
                    if length > max_length:
                        max_length = length
                        critical_path = path

        return critical_path

    def _calculate_path_length(self, path: List[int]) -> float:
        """Рассчитывает длину пути"""
        length = sum(self.operations[node].complexity for node in path)
        length += sum(self.graph[path[i]][path[i+1]]['data_volume']
                     for i in range(len(path)-1))
        return length

    def print_info(self):
        """Вывод информации о графе задач"""
        print(f"\nTask Graph Type {self.graph_type} Properties:")
        print(f"Number of tasks: {self.graph.number_of_nodes()}")
        print(f"Number of dependencies: {self.graph.number_of_edges()}")
        print(f"Total computational complexity: "
              f"{sum(self.operations[n].complexity for n in self.graph.nodes())}")
        print(f"Total data transfer: "
              f"{sum(d['data_volume'] for (u,v,d) in self.graph.edges(data=True))}")

        critical_path = self.find_critical_path()
        if critical_path:
            print("\nCritical Path Information:")
            print(f"Path: {' -> '.join(map(str, critical_path))}")
            print(f"Length: {self._calculate_path_length(critical_path)}")

    def get_task_by_id(self, task_id: int) -> Task:
        """
        Возвращает задачу по ее ID
        """
        for task in self.operations:
            if task.id == task_id:
                return task
        raise ValueError(f"Задача с ID {task_id} не найдена")


@dataclass
class NodeStats:
    compute_load: float = 0.0    # Вычислительная нагрузка в операциях
    receive_load: float = 0.0    # Нагрузка на прием в байтах
    send_load: float = 0.0       # Нагрузка на отправку в байтах
    end_time: float = 0.0

class TaskScheduler:
    """
    Планировщик задач для распределенной системы.
    Управляет распределением задач по узлам сети и планирует передачу данных между ними.
    """
        
    def __init__(self, task_graph: TaskGraph, net_graph: NetGraph):
        """
        Инициализация планировщика задач.
        
        Параметры:
        ----------
        task_graph : TaskGraph
            Граф задач с весами операций и зависимостями
        net_graph : NetGraph
            Граф сети с характеристиками узлов и каналов связи
        """

        self.task_graph = task_graph
        self.net_graph = net_graph

    def get_edge_speed(self, node1, node2):
        """
        Получает скорость передачи данных между узлами из графа сети
        
        Параметры:
        ----------
        node1, node2 : int
            Номера узлов между которыми определяется скорость
            
        Возвращает:
        -----------
        float
            Скорость передачи данных между узлами
        """
        try:
            return self.net_graph.graph[node1][node2]['bandwidth']  
        except:
            return self.net_graph.net_speed  # Возвращаем дефолтную скорость если не задана

    def assign_tasks_to_nodes(self, distribution):
        """
        Создает словарь соответствия задач узлам сети.
        
        Параметры:
        ----------
        distribution : list
            Список распределения задач по узлам
            
        Возвращает:
        -----------
        dict
            Словарь {номер_задачи: номер_узла}
        """
        return {task: distribution[task] for task in range(len(distribution))}

    def shortest_path(self, start, end):
        """
        Находит кратчайший путь между узлами в графе сети.
        
        Параметры:
        ----------
        start : int
            Начальный узел
        end : int
            Конечный узел
            
        Возвращает:
        -----------
        list
            Список узлов, составляющих кратчайший путь
        """
        return nx.shortest_path(self.net_graph.graph, start, end)

    def calculate_schedule(self, distribution: list):
        """
        Рассчитывает расписание выполнения задач и передачи данных.
        
        Параметры:
        ----------
        distribution : list
            Список распределения задач по узлам
            
        Основные этапы:
        1. Инициализация структур данных для хранения расписания
        2. Обход задач в топологическом порядке
        3. Планирование выполнения каждой задачи
        4. Планирование передачи данных между зависимыми задачами
        """
        # Инициализация структур данных
        self.node_assignments = self.assign_tasks_to_nodes(distribution)
        self.schedule = defaultdict(list)  # Расписание для каждого узла
        self.data_transfers = []  # Список передач данных
        current_time = defaultdict(float)  # Текущее время для каждого узла

        # Обход задач в топологическом порядке
        for task in nx.topological_sort(self.task_graph.graph):
            node = self.node_assignments[task]
            
            # Расчет времени выполнения задачи
            start_time = current_time[node]
            duration = self.task_graph.operations[task].get_task_complexity() / self.net_graph.nodes[node].get_node_performance()
            end_time = start_time + duration

            # Добавление задачи в расписание
            self.schedule[node].append((task, start_time, end_time, 'task'))
            current_time[node] = end_time
            
            # Обработка передачи данных преемникам
            for successor in self.task_graph.graph.successors(task):
                successor_node = self.node_assignments[successor]
                
                # Если задачи на разных узлах - планируем передачу данных
                if successor_node != node:
                    data_volume = self.task_graph.graph[task][successor]['data_volume']
                    path = self.shortest_path(node, successor_node)
                    
                    # Обработка каждого узла в пути передачи
                    for i in range(len(path)):
                        current_node = path[i]
                        
                        # Прием данных (кроме начального узла)
                        if i > 0:
                            receive_time = data_volume / self.net_graph.nodes[current_node].get_node_performance()
                            receive_start = current_time[current_node]
                            receive_end = receive_start + receive_time
                            self.schedule[current_node].append((f"Receive T{task}->{successor}", receive_start, receive_end, 'receive'))
                            current_time[current_node] = receive_end

                        # Отправка данных (кроме конечного узла)
                        if i < len(path) - 1:
                            send_time = data_volume / self.net_graph.nodes[current_node].get_node_performance()
                            send_start = current_time[current_node]
                            send_end = send_start + send_time
                            self.schedule[current_node].append((f"Send T{task}->{successor}", send_start, send_end, 'send'))
                            current_time[current_node] = send_end

                        # Передача данных по сети (кроме последнего узла)
                        if i < len(path) - 1:
                            next_node = path[i + 1]
                            edge_speed = self.get_edge_speed(current_node, next_node)
                            network_transfer_time = data_volume / edge_speed
                            transfer_start = max(current_time[current_node], current_time[next_node])
                            transfer_end = transfer_start + network_transfer_time
                            self.data_transfers.append((current_node, next_node, transfer_start, transfer_end, task, successor))
                            current_time[current_node] = transfer_end
                            current_time[next_node] = transfer_end

    def get_total_execution_time(self):
        end_times = [max(tasks, key=lambda x: x[2])[2] for tasks in self.schedule.values()]
        return max(end_times)

    def create_gantt_chart(self):
        """
        Создает диаграмму Ганта для визуализации расписания.
        
        Отображает:
        - Выполнение задач на узлах
        - Операции отправки/получения данных
        - Передачу данных между узлами
        - Простои узлов
        """
        # Создание фигуры с увеличенным размером и отступами
        fig, ax = plt.subplots(figsize=(25.6, 14.4), dpi=100)
        plt.subplots_adjust(bottom=0.2)  # Увеличиваем отступ снизу для легенды

        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300

        # Настройка цветов
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.task_graph.graph.nodes())))
        send_color = 'lightblue'
        receive_color = 'lightgreen'
        transfer_color = 'lightgrey'
        idle_color = 'white'

         # Настройка цветов для задач (исключая служебные цвета)
        task_colors = [
            '#FF8C00',  # Dark Orange
            '#4169E1',  # Royal Blue
            '#8B008B',  # Dark Magenta
            '#2E8B57',  # Sea Green
            '#800000',  # Maroon
            '#4B0082',  # Indigo
            '#556B2F',  # Dark Olive Green
            '#8B4513',  # Saddle Brown
            '#483D8B',  # Dark Slate Blue
            '#008080',  # Teal
            '#9932CC',  # Dark Orchid
            '#B8860B',  # Dark Goldenrod
            '#006400',  # Dark Green
            '#8B0000',  # Dark Red
            '#191970',  # Midnight Blue
            '#BDB76B',  # Dark Khaki
            '#FF4500',  # Orange Red
            '#00CED1',  # Dark Turquoise
            '#9400D3',  # Dark Violet
            '#696969'   # Dim Gray
        ]

        # Подготовка осей
        y_ticks = []
        y_labels = []
        total_time = self.get_total_execution_time()
    
        node_height = 0.4  # Уменьшаем высоту полосы узла
        transfer_height = 0.1  # Уменьшаем высоту полосы передачи
        node_spacing = 1.0  # Увеличиваем расстояние между узлами

        # Отрисовка для каждого узла
        for i, node in enumerate(range(len(self.net_graph.nodes))):
            y_pos = i * node_spacing  # Позиция узла с учетом интервала
            y_ticks.append(y_pos)
            y_labels.append(f"Node {node}")
            
            # Фоновая полоса (простой)
            ax.barh(y_pos, total_time, left=0, height=node_height,
                    align='center', color=idle_color, alpha=0.3)

            if node in self.schedule:
                for task, start, end, task_type in self.schedule[node]:
                    duration = end - start

                    if task_type == 'task':
                        # Отрисовка выполнения задачи
                        task_color = task_colors[int(task) % len(task_colors)]
                        bar = ax.barh(y_pos, duration, left=start, height=node_height,
                            align='center', color=task_color, alpha=0.8)
                        
                        if duration > total_time * 0.05:
                            text_x = start + duration/2
                            ax.text(text_x, y_pos, f'T{task}',
                                ha='center', va='center',
                                fontsize=10, fontweight='bold',
                                bbox=dict(facecolor='white', alpha=0.7,
                                        edgecolor='none', pad=1))
                            
                    elif task_type == 'send':
                        # Отрисовка отправки данных
                        bar = ax.barh(y_pos - node_height/2, duration, left=start, 
                            height=transfer_height,
                            align='center', color=send_color, alpha=0.7)
                        if duration > total_time * 0.08:
                            task_label = task.replace('Send T', 'S').replace('->', '→')
                            ax.text(start + duration/2, y_pos - node_height/2, task_label,
                                ha='center', va='center', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.7,
                                        edgecolor='none', pad=1))

                    elif task_type == 'receive':
                        # Отрисовка получения данных
                        bar = ax.barh(y_pos + node_height/2, duration, left=start, 
                            height=transfer_height,
                            align='center', color=receive_color, alpha=0.7)
                        if duration > total_time * 0.08:
                            task_label = task.replace('Receive T', 'R').replace('->', '→')
                            ax.text(start + duration/2, y_pos + node_height/2, task_label,
                                ha='center', va='center', fontsize=8,
                                bbox=dict(facecolor='white', alpha=0.7,
                                        edgecolor='none', pad=1))

        # Отрисовка передач данных между узлами как кривые линии
        for src, dst, start, end, task, successor in self.data_transfers:
            y_start = src * node_spacing
            y_end = dst * node_spacing
            duration = end - start
            mid_time = start + duration/2
            
            # Создаем кривую линию между узлами
            curve = PathPatch(
                Path([
                    (start, y_start),  # Начальная точка
                    (mid_time, (y_start + y_end)/2),  # Контрольная точка
                    (end, y_end)  # Конечная точка
                ],
                [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                facecolor='none',
                edgecolor=transfer_color,
                alpha=0.6,
                linestyle='-',
                linewidth=2
            )
            ax.add_patch(curve)
            
            # Добавляем текст с информацией о передаче
            if duration > total_time * 0.05:
                ax.text(mid_time, (y_start + y_end)/2,
                    f'T{task}→T{successor}',
                    ha='center', va='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7,
                            edgecolor='none', pad=1))

            # Настройка осей
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=10)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_title('Task Execution and Data Transfer Schedule', 
                        fontsize=14, pad=20)
            # Настройка сетки
            ax.grid(True, axis='x', alpha=0.5, linestyle='--')

            # Создаем легенду
            legend_elements = [
                mpatches.Patch(color=colors[0], alpha=0.8, label='Task Execution'),
                mpatches.Patch(color=send_color, alpha=0.5, label='Data Send'),
                mpatches.Patch(color=receive_color, alpha=0.5, label='Data Receive'),
                mpatches.Patch(color=transfer_color, alpha=0.5, label='Data Transfer'),
                mpatches.Patch(color=idle_color, alpha=0.3, label='Idle Time')
            ]
            
                # Размещение легенды внизу с несколькими колонками
            ax.legend(handles=legend_elements, 
                    loc='center', 
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=5,  # Количество колонок
                    fontsize=10,
                    frameon=True,
                    fancybox=True,
                    shadow=True)

        # Оптимизация компоновки
        plt.tight_layout()
        
        # Регулировка отступов для предотвращения обрезания
        plt.margins(x=0.02)
        
        plt.show()

    def get_complete_analysis(self, distribution: list):
        """
        Полный анализ системы с учетом реальной нагрузки и временных характеристик
        """

        self.calculate_schedule(distribution)

        analysis = {
            'nodes': {
                node_id: {
                    'performance': self.net_graph.nodes[node].get_node_performance(),
                    'compute_load': 0.0,  # в операциях
                    'data_received': 0.0, # в байтах
                    'data_sent': 0.0,     # в байтах
                    'working_time': 0.0,
                    'send_time': 0.0,
                    'receive_time': 0.0
                } 
                for node_id, node in enumerate(self.net_graph.nodes)
            },
            'tasks': {
                task_id: {
                    'complexity': self.task_graph.operations[task].get_task_complexity(),
                    'assigned_node': self.node_assignments.get(task_id),
                    'execution_time': 0.0
                }
                for task_id, task in enumerate(self.task_graph.operations)
            },
            'transfers': [],
            'statistics': {
                'total_time': 0.0,
                'total_operations': 0.0,
                'total_data_transferred': 0.0,
                'transfer_count': 0
            }
        }

        # Анализ назначения и выполнения задач
        for task_id, node_id in self.node_assignments.items():
            task_complexity = self.task_graph.operations[task_id].get_task_complexity()
            analysis['nodes'][node_id]['compute_load'] += task_complexity
            analysis['statistics']['total_operations'] += task_complexity

        # Анализ расписания и передач данных
        for node_id, tasks in self.schedule.items():
            node_stats = analysis['nodes'][node_id]
            
            for task_name, start, end, task_type in tasks:
                duration = end - start
                
                if task_type == 'task':
                    node_stats['working_time'] += duration
                elif task_type == 'send':
                    node_stats['send_time'] += duration
                elif task_type == 'receive':
                    node_stats['receive_time'] += duration

        # Анализ передач данных
        for src, dst, start, end, task, successor in self.data_transfers:
            data_volume = self.task_graph.graph[task][successor]['data_volume']
            
            analysis['nodes'][src]['data_sent'] += data_volume
            analysis['nodes'][dst]['data_received'] += data_volume
            analysis['statistics']['total_data_transferred'] += data_volume
            
            transfer_info = {
                'from_task': task,
                'to_task': successor,
                'from_node': src,
                'to_node': dst,
                'data_volume': data_volume,
                'start_time': start,
                'end_time': end
            }
            analysis['transfers'].append(transfer_info)

        analysis['statistics']['transfer_count'] = len(self.data_transfers)
        analysis['statistics']['total_time'] = (
            max(
                max(end for _, _, end, _ in tasks)
                for tasks in self.schedule.values()
                if tasks
            )
            if self.schedule else 0
        )

        return analysis

    def print_complete_analysis(self):
        """
        Вывод полного анализа системы
        """
        analysis = self.get_complete_analysis()
        
        print("\nАНАЛИЗ СИСТЕМЫ")
        print("=" * 50)
        
        # Информация об узлах
        print("\nХарактеристики и нагрузка узлов:")
        for node_id, node_info in analysis['nodes'].items():
            print(f"\nУзел {node_id}:")
            print(f"  Производительность: {node_info['performance']} оп/с")
            print(f"  Вычислительная нагрузка: {node_info['compute_load']} операций")
            print(f"  Принято данных: {node_info['data_received']} байт")
            print(f"  Отправлено данных: {node_info['data_sent']} байт")
            print(f"  Время работы: {node_info['working_time']:.2f}")
            print(f"  Время на передачу: {node_info['send_time']:.2f}")
            print(f"  Время на прием: {node_info['receive_time']:.2f}")

        # Информация о задачах
        print("\nИнформация о задачах:")
        for task_id, task_info in analysis['tasks'].items():
            print(f"Задача {task_id}:")
            print(f"  Сложность: {task_info['complexity']} операций")
            print(f"  Назначена на узел: {task_info['assigned_node']}")

        # Общая статистика
        stats = analysis['statistics']
        print("\nОбщая статистика:")
        print(f"Общее время выполнения: {stats['total_time']:.2f}")
        print(f"Общее количество операций: {stats['total_operations']}")
        print(f"Общий объем переданных данных: {stats['total_data_transferred']} байт")
        print(f"Количество передач данных: {stats['transfer_count']}")


class NetworkOptimizationProblem(OptimizationProblem):
    """
    Класс для решения задачи оптимизации распределения задач в сети.
    
    Attributes:
        network_graph (NetGraph): Граф сети
        task_graph (TaskGraph): Граф задач
        t_lim (float): Временное ограничение
        net_speed (float): Скорость сети по умолчанию
        scheduler (TaskScheduler): Планировщик задач
    """
    def __init__(self, 
                 network_graph: NetGraph,
                 task_graph: TaskGraph,
                 f_objective: list,
                 constraints: list = None,
                 bounds: dict = None,
                 dtype: type = int,
                 t_lim: float = 5,
                 net_speed: float = 1000,
                 name: str = "NETproblem_1",
                 node_functions: list = None,
                 function_constraints: list = None,
                 special_function_constraints: list = None):
        """
        Инициализация задачи оптимизации сети.

        Args:
            network_graph: Граф сети
            task_graph: Граф задач
            f_objective: Целевые функции оптимизации
            constraints: Функции-ограничения для распределения
            bounds: Ограничения на распределение задач по узлам
            dtype: Тип данных для значений
            t_lim: Временное ограничение
            net_speed: Скорость сети по умолчанию
            name: Имя задачи
            node_functions: Функции узлов
            function_constraints: Ограничения на функции
            special_function_constraints: Специальные ограничения
        """

        # Инициализация базовых параметров
        self.network_graph = network_graph
        self.task_graph = task_graph
        self.t_lim = t_lim
        self.net_speed = net_speed
        
        # Определение размерности вектора решения
        vector_length = task_graph.graph.number_of_nodes()
        
        # Формирование ограничений на распределение
        bounds = self._create_constraints(bounds, vector_length)
        
        # Инициализация родительского класса
        super().__init__(
            f_objective=f_objective,
            constraints=constraints,
            bounds=bounds,
            dtype=dtype,
            len=vector_length,
            name=name,
            node_functions=node_functions,
            function_constraints=function_constraints,
            special_function_constraints=special_function_constraints
        )
        
        # Создание планировщика задач
        self.scheduler = TaskScheduler(task_graph, network_graph)

    def _create_constraints(self, bounds: dict, vector_length: int) -> list:
        """
        Создает список ограничений для распределения задач.

        Args:
            bounds: Словарь с ограничениями для конкретных узлов
            vector_length: Длина вектора решения

        Returns:
            list: Список кортежей с ограничениями (min, max)
        """
        # Базовые ограничения для всех узлов
        max_node = self.network_graph.graph.number_of_nodes() - 1
        default_constraints = [(0, max_node) for _ in range(vector_length)]
        
        # Применение специальных ограничений если они есть
        if bounds:
            for node, constraint in bounds.items():
                default_constraints[node] = constraint
                
        return default_constraints

    def validate_solution(self, solution: list) -> bool:
        """
        Проверяет допустимость решения.

        Args:
            solution: Вектор распределения задач

        Returns:
            bool: True если решение допустимо, False иначе
        """
        try:
            self.scheduler.calculate_schedule(solution)
            return True
        except Exception:
            return False

    def get_solution_metrics(self, solution: list) -> dict:
        """
        Вычисляет метрики для данного решения.

        Args:
            solution: Вектор распределения задач

        Returns:
            dict: Словарь с метриками решения
        """
        self.scheduler.calculate_schedule(solution)
        return {
            'total_time': self.scheduler.get_total_execution_time(),
            'transfer_count': self.scheduler.get_transfer_count(),
            'send_times': self.scheduler.get_node_send_times(),
            'working_times': self.scheduler.get_node_working_times()
        }

    def net_status(self, solution: list):
        """
        Выводит подробную информацию о состоянии сети в красивом форматировании.
        
        Параметры:
        ----------
        scheduler : TaskScheduler
            Планировщик задач с информацией о сети
        """

        self.scheduler.calculate_schedule(solution)

        # Цвета для форматирования
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        
        def print_separator(char="=", length=50):
            print(BLUE + char * length + ENDC)
            
        def print_section_header(text):
            print(HEADER + BOLD + f"\n{text:^50}" + ENDC)
            print_separator()
        
        # Получаем статистику
        stats = self.scheduler.get_timing_statistics()
        
        # Заголовок
        print_section_header("СОСТОЯНИЕ СЕТИ")
        
        # Информация о узлах
        print_section_header("ХАРАКТЕРИСТИКИ УЗЛОВ")
        for i, node in enumerate(self.scheduler.net_graph.nodes):
            print(f"{GREEN}Узел {i}:{ENDC}")
            print(f"├─ Вычислительная мощность: {BOLD}{node.p:.4f}{ENDC}")
            print(f"├─ Время работы: {BOLD}{stats['working_times'].get(i, 0):.4f}{ENDC}")
            print(f"├─ Время отправки данных: {BOLD}{stats['send_times'].get(i, 0):.4f}{ENDC}")
            print(f"└─ Время приема данных: {BOLD}{stats['receive_times'].get(i, 0):.4f}{ENDC}")
        
        # Информация о задачах
        print_section_header("РАСПРЕДЕЛЕНИЕ ЗАДАЧ")
        current_node = None
        for task, node in sorted(self.scheduler.node_assignments.items(), key=lambda x: x[1]):
            if current_node != node:
                if current_node is not None:
                    print(f"└{'─' * 30}")
                current_node = node
                print(f"\n{GREEN}Узел {node}:{ENDC}")
            workload = self.scheduler.task_graph.operations[task].get_task_complexity()
            print(f"├─ Задача {task} (сложность: {workload})")
        
        # Информация о передачах данных
        print_section_header("ПЕРЕДАЧИ ДАННЫХ")
        if self.scheduler.data_transfers:
            for src, dst, start, end, task, successor in self.scheduler.data_transfers:
                duration = end - start
                print(f"{BLUE}Передача T{task}→T{successor}:{ENDC}")
                print(f"├─ Маршрут: Узел {src} → Узел {dst}")
                print(f"├─ Время начала: {start:.2f}")
                print(f"├─ Время окончания: {end:.2f}")
                print(f"└─ Длительность: {duration:.2f}")
        else:
            print(f"{WARNING}Нет передач данных между узлами{ENDC}")
        
        # Общая статистика
        print_section_header("ОБЩАЯ СТАТИСТИКА")
        print(f"Общее время выполнения: {BOLD}{stats['total_time']:.2f}{ENDC}")
        print(f"Количество передач: {BOLD}{stats['transfer_count']}{ENDC}")
        
        # Загрузка узлов
        print_section_header("ЗАГРУЗКА УЗЛОВ")
        for node in range(len(self.scheduler.net_graph.nodes)):
            total_time = stats['total_time']
            working_time = stats['working_times'].get(node, 0)
            utilization = (working_time / total_time) * 100 if total_time > 0 else 0
            
            # Создаем визуальную шкалу загрузки
            bar_length = 20
            filled_length = int(utilization / 100 * bar_length)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Выбираем цвет в зависимости от загрузки
            if utilization < 30:
                color = FAIL
            elif utilization < 70:
                color = WARNING
            else:
                color = GREEN
                
            print(f"Узел {node}: {color}{bar}{ENDC} {utilization:.1f}%")
        
        print_separator("=", 50)

        print(self.get_info(solution))