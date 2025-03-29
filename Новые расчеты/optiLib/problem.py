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
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union
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
            **{f.__name__: f(vector, self) for f in self.f_objective},
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
    e_receive: float            = 0.0  # Энергозатраты на прием
    e_comp: float               = 0.0  # Энергозатраты на вычисления
    e_send: float               = 0.0  # Энергозатраты на отправку
    cost: float                 = 0.0  # Стоимость устройства
    operation_cost: float       = 0.0  # Стоимость устройства
    failure_rate: float         = 0.0  # Интенсивность отказов

    def __str__(self) -> str:
                return (f"Node {self.id} - Perf: {self.performance}, "
                f"Cost: {self.cost:.2f}, Fail rate: {self.failure_rate:.2e}")

    def get_node_performance(self) -> float:
        """
        Возвращает производительность узла
        """
        # Этот метод должен быть реализован в зависимости от вашей структуры сети
        return self.performance


class NetGraph:
    def __init__(self, graph_type: int = 1,
                 nodes_params: Dict[int, Dict] = None,
                 edges: List[Tuple[int, int, float]] = None,
                 matrix: np.ndarray = None,
                 net_power: Tuple[int, int] = (100, 2500)) -> None:
        """
        Инициализация графа сети
        
        Args:
            graph_type: Тип графа (1 - сложный, 2 - линейный, 3 - пользовательский)
            nodes_params: Параметры узлов для пользовательского графа
            edges: Рёбра для пользовательского графа (формат: [(u, v, bandwidth)])
            matrix: Матрица смежности (если None, используется предустановленная топология)
            net_power: Диапазон производительности для случайной генерации
        """
        self.graph_type = graph_type
        
        if graph_type == 3:
            if nodes_params is None or edges is None:
                raise ValueError("Для пользовательского графа нужны nodes_params и edges")
            self.graph, self.nodes = self._create_custom_network(nodes_params, edges)
        elif matrix is not None:
            self.graph = nx.Graph(matrix)
            self.nodes = self._create_nodes_from_matrix(net_power)
        else:
            self.graph, self.nodes = self._create_predefined_network()

    def _create_custom_network(self, nodes_params: Dict[int, Dict], 
                             edges: List[Tuple]) -> Tuple[nx.Graph, Dict[int, NetworkNode]]:
        """Создание пользовательской сети"""
        network = nx.Graph()
        nodes = {}
        # Создание узлов
        for node_id, node_data in nodes_params.items():
            if isinstance(node_data, NetworkNode):
                # Используем существующий объект
                current_node = node_data
            else:
                # Создаем новый NetworkNode из параметров
                current_node = NetworkNode(
                    id=node_id,
                    performance=node_data.get("performance", 0.0),
                    e_receive=node_data.get("e_receive", 0.0),
                    e_comp=node_data.get("e_comp", 0.0),
                    e_send=node_data.get("e_send", 0.0),
                    cost=node_data.get("cost", 0.0),
                    operation_cost=node_data.get("operation_cost", 0.0),
                    failure_rate=node_data.get("failure_rate", 0.0)
                )
            # Добавляем в результат и в граф
            nodes[node_id] = current_node
            network.add_node(node_id)
        
        # Добавление рёбер
        for u, v, bandwidth in edges:
            network.add_edge(u, v, bandwidth=bandwidth)
        
        return network, nodes

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
            0: 100, 1: 100, 2: 100, 3: 100,
            4: 100, 5: 200, 6: 100, 7: 100, 8: 200
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
                cost=perf * np.random.uniform(0.8, 1.2),
                operation_cost = perf * np.random.uniform(0.8, 1.2)* np.random.uniform(0.2, 1.2)
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
        plt.figure(figsize=(15, 10))
        ax = plt.gca()

        # Определение позиций узлов
        if self.graph_type == 1:
            pos = self._complex_layout()
        elif self.graph_type == 2:
            pos = {i: (i, 0) for i in self.nodes}
        else:
            pos = nx.spring_layout(self.graph, seed=42)  # Для пользовательских графов

        self._draw_network(ax, pos)

    def _complex_layout(self):
        """Предопределенные позиции для сложной сети"""
        return {
            0: (-2, 0), 1: (-1, -1), 2: (-1, 1),
            3: (0, -2), 4: (0, 0), 5: (0, 2),
            6: (1, 3), 7: (1, 1), 8: (1, -1)
        }

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

    def _draw_network(self, ax, pos):
        """Отрисовка сети"""
        node_colors = [self.nodes[n].performance for n in self.graph.nodes()]
        nodes_draw = nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=1500,
            cmap=plt.cm.plasma,
            ax=ax
        )
        
        nx.draw_networkx_edges(
            self.graph, pos,
            width=2,
            edge_color='gray',
            alpha=0.6
        )
        
        edge_labels = nx.get_edge_attributes(self.graph, 'bandwidth')
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=10
        )
        
        labels = {n: f"N{n}\n{self.nodes[n].performance}" for n in self.graph.nodes()}
        nx.draw_networkx_labels(
            self.graph, pos,
            labels=labels,
            font_size=10
        )
        
        plt.colorbar(nodes_draw, ax=ax, label='Performance (units)')
        plt.title(f"Network Topology - Type {self.graph_type}", fontsize=14)
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
        for nodeid, node in self.nodes.items():
            if node.id == node_id:
                return node
        raise ValueError(f"Узел с ID {node_id} не найден")    

    def get_node_performance(self, node_id: int):
        return self.get_node_by_id(node_id).get_node_performance()



@dataclass
class Task:
    id: int
    complexity: float  # Вычислительная сложность
    input_data: float  # Объем входных данных
    output_data: float # Объем выходных данных

    def get_task_complexity(self) -> float:
        """
        Возвращает производительность узла
        """
        # Этот метод должен быть реализован в зависимости от вашей структуры сети
        return self.complexity
    

class TaskGraph:
    def __init__(self, graph_type: int = 1, 
                 tasks_params: Dict[int, Dict] = None, 
                 edges: List[Tuple[int, int, float]] = None):
        """
        Инициализация графа задач

        Args:
            graph_type: Тип графа (1 - сложный, 2 - простой, 3 - полносвязный)
        """
        self.graph_type = graph_type
        if tasks_params is not None and edges is not None:
            self.graph, self.operations = self._create_custom_task_graph(tasks_params, edges)
        else:
            self.graph, self.operations = self._create_task_graph()

    def _create_task_graph(self) -> Tuple[nx.DiGraph, Dict[int, Task]]:
        """Создание графа задач выбранного типа"""
        if self.graph_type == 1:
            return self._create_complex_task_graph()
        elif self.graph_type == 2:
            return self._create_simple_task_graph()
        elif self.graph_type == 3:
            return self._create_fully_connected_task_graph()
        else:
            raise ValueError("Неизвестный тип графа")

    def _create_custom_task_graph(self, tasks_params: Dict[int, Dict], 
                                edges: List[Tuple]) -> Tuple[nx.DiGraph, Dict[int, Task]]:
        """Создание пользовательского графа задач"""
        task_graph = nx.DiGraph()
        return self._create_graph_structure(task_graph, tasks_params, edges)

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
                output_data=0.0 # Будет обновлено после добавления ребер 
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

        # Определяем позиции узлов
        if self.graph_type in [1, 2, 3]:
            pos = self._get_predefined_positions()
        else:
            pos = nx.spring_layout(self.graph, seed=42)  # Автоматическое позиционирование

        self._draw_graph(ax, pos)

    def _get_predefined_positions(self):
        """Возвращает предопределенные позиции для стандартных графов"""
        if self.graph_type == 1:
            return {
                0: (-2, 0), 1: (0, 2), 2: (2, 2), 3: (4, 2),
                4: (0, 0), 5: (2, 0), 6: (4, 0),
                7: (0, -2), 8: (2, -2), 9: (6, 0)
            }
        elif self.graph_type == 2:
            return {
                0: (-2, 0),
                1: (0, 2), 2: (0, 0), 3: (0, -2),
                4: (2, 0)
            }
        else:  # type 3
            return {
                0: (-1, 1), 1: (1, 1),
                2: (-1, -1), 3: (1, -1)
            }

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
    Планировщик задач для распределенных систем с поддержкой трех моделей передачи данных.
    Обеспечивает:
    - Распределение задач по узлам сети
    - Планирование выполнения задач с учетом зависимостей
    - Управление передачей данных между узлами
    - Визуализацию расписания через диаграммы Ганта
    - Подробный анализ производительности системы
    
    Основные модели работы:
    1. Прямая передача данных между узлами
    2. Модель Барского (передача через магистраль)
    3. Модель Топоркова (передача через промежуточные узлы)
    
    Типовая последовательность использования:
    1. Создать экземпляр класса с нужной моделью
    2. Вызвать calculate_schedule() с распределением задач
    3. Получить результаты через get_complete_analysis()
    4. Визуализировать через create_gantt_chart()
    """
        
    def __init__(self, task_graph, net_graph, 
                 model_type: int = 1,
                 original_net_graph = None):
        """
        Инициализация планировщика
        
        Args:
            task_graph (TaskGraph): Граф задач с зависимостями
            net_graph (NetGraph): Граф сети с характеристиками узлов
            model_type (int): 
                1 - Прямая передача данных
                2 - Модель Барского (магистраль)
                3 - Модель Топоркова (промежуточные узлы)
            intermediate_node_params (dict): Параметры магистрали для model_type=2
                Пример: {'performance': 1000, 'cost': 0.5}
            intermediate_nodes_params (list): Список параметров промежуточных узлов для model_type=3
                Пример: [{'performance': 500}, {'performance': 800}]
            channel_types (dict): Типы каналов для model_type=3
                Пример: {'wifi': {'bandwidth_multiplier': 1.5, 'cost_multiplier': 1.2}}
            bandwidth_factor (float): Множитель пропускной способности для model_type=2
        
        Пример:
            scheduler = TaskScheduler(
                task_graph=my_task_graph,
                net_graph=my_net_graph,
                model_type=2,
                intermediate_node_params={'performance': 2000}
            )
        """
        self.task_graph = task_graph
        self.original_net_graph = original_net_graph
        self.model_type = model_type
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

    def calculate_schedule(
            self, 
            distribution: List[int], 
            custom_paths: Optional[Dict[Tuple[int, int], List[int]]] = None,
            allow_parallel: bool = False
        ) -> Tuple[Dict[int, List[Tuple]], List[Tuple]]:
        """
        Рассчитывает расписание выполнения задач и передач данных с учетом:
        - Зависимостей задач
        - Топологии сети
        - Специфики моделей (магистрали, промежуточные узлы)
        
        Параметры:
        ----------
        distribution : list[int]
            Список, где индекс - ID задачи, значение - ID назначаемого узла
        custom_paths : dict, optional
            Пользовательские пути передачи данных в формате { (task_from, task_to): [node_path] }
        allow_parallel : bool
            Если True, разрешает параллельные операции на узле (кроме промежуточных)
        
        Возвращает:
        -----------
        schedule : dict
            Расписание для каждого узла: {node_id: [(название_операции, start, end, тип), ...]}
        data_transfers : list
            Список передач данных: [(src_node, dst_node, start, end, task_from, task_to), ...]
        """
        
        self.schedule = defaultdict(list)  # Сброс расписания
        self.data_transfers = []  # Сброс списка передач
        self.current_time = defaultdict(float)  # Сброс времени
        
        # Инициализация назначения задач
        self.node_assignments = self.assign_tasks_to_nodes(distribution)
        
        # Инициализация: Если custom_paths не предоставлен, используем пустой словарь
        if custom_paths is None:
            custom_paths = {}

        # Инициализация расписания "бездействия" для всех узлов
        for node in self.net_graph.nodes:
            self.schedule[node].append(('Idle', 0.0, 0.0, 'idle'))

        # Обход задач в топологическом порядке (от корневых к листьям)
        for task in nx.topological_sort(self.task_graph.graph):
            node = self.node_assignments[task]
            start_time = self.current_time[node]

            # Расчет времени выполнения задачи
            task_duration = (
                self.task_graph.operations[task].get_task_complexity() 
                / self.net_graph.get_node_performance(node)
            )
            end_time = start_time + task_duration

            # Добавление задачи в расписание
            self.schedule[node].append((task, start_time, end_time, 'task'))
            self._update_time(node, end_time, allow_parallel)  # Обновление времени узла

            # Обработка зависимостей: для каждой задачи-наследника
            for successor in self.task_graph.graph.successors(task):
                successor_node = self.node_assignments[successor]
                if successor_node != node:  # Требуется передача данных
                    data_volume = self.task_graph.graph[task][successor]['data_volume']
                    
                    # Определение пути:
                    # 1. Использовать пользовательский путь, если он указан
                    # 2. Иначе - кратчайший путь
                    path = custom_paths.get(
                        (task, successor), 
                        self.get_shortest_path(node, successor_node)
                    )

                    # Передача не может начаться раньше завершения задачи
                    transfer_start = end_time

                    # Обработка передачи данных
                    transfer_info = self._process_data_transfer(
                        path, 
                        data_volume, 
                        transfer_start, 
                        allow_parallel, 
                        task, 
                        successor
                    )

                    # Обновление времени для конечного узла
                    last_node = path[-1]
                    self.current_time[last_node] = max(
                        self.current_time[last_node], 
                        transfer_info['end_time']
                    )

        # Сортировка событий и заполнение промежутков
        for node in self.schedule:
            self.schedule[node].sort(key=lambda x: x[1])
            self._fill_gaps(node)  # Добавление событий "бездействия"

        return dict(self.schedule), self.data_transfers

    def _process_data_transfer(
            self,
            path: List[int],
            data_volume: float,
            initial_start: float,
            allow_parallel: bool,
            task: int,
            successor: int
        ) -> Dict:
        """
        Обрабатывает передачу данных по заданному пути

        Параметры:
        ----------
        path : list[int]
            Путь передачи: список узлов от источника к получателю
        data_volume : float
            Объем передаваемых данных
        initial_start : float
            Минимальное время начала передачи (не раньше этого момента)
        allow_parallel : bool
            Разрешение параллельных операций на узлах
        task : int
            ID исходной задачи
        successor : int
            ID целевой задачи

        Возвращает:
        -----------
        transfer_info : dict
            {'start_time': ..., 'end_time': ...}
        """
        transfer_info = {
            'start_time': initial_start,
            'end_time': initial_start
        }
        
        # Время, когда данные доступны на каждом узле пути
        data_arrival_time = {path[0]: initial_start}
        
        # Проходим по всем парам соседних узлов в пути
        for i in range(len(path) - 1):
            src_node = path[i]
            dst_node = path[i+1]
            is_src_intermediate = src_node not in self.original_net_graph.nodes
            
            # 1. Отправка данных с узла src_node
            send_start = max(self.current_time[src_node], data_arrival_time[src_node])
            send_duration = data_volume / self.net_graph.get_node_performance(src_node)
            send_end = send_start + send_duration
            
            # Добавляем операцию отправки в расписание
            self.schedule[src_node].append((f"Send_{task}_{successor}", send_start, send_end, 'send'))
            
            # Обновляем время узла, если он не промежуточный и не разрешены параллельные операции
            if not is_src_intermediate and not allow_parallel:
                self._update_time(src_node, send_end, allow_parallel)
            
            # 2. Передача данных по сети между узлами
            edge_speed = self.get_edge_speed(src_node, dst_node)
            network_time = data_volume / edge_speed
            transfer_start = send_end  # Передача начинается сразу после завершения отправки
            transfer_end = transfer_start + network_time
            
            # Добавляем информацию о передаче данных
            self.data_transfers.append((
                src_node,
                dst_node,
                transfer_start,
                transfer_end,
                task,
                successor
            ))
            
            # Данные доступны на следующем узле после завершения передачи
            data_arrival_time[dst_node] = transfer_end
        
        # 3. Прием данных на последнем узле
        last_node = path[-1]
        is_last_intermediate = last_node not in self.original_net_graph.nodes
        
        receive_start = data_arrival_time[last_node]  # Начало приема после прибытия данных
        receive_duration = data_volume / self.net_graph.get_node_performance(last_node)
        receive_end = receive_start + receive_duration
        
        # Добавляем операцию приема в расписание последнего узла
        self.schedule[last_node].append((f"Receive_{task}_{successor}", receive_start, receive_end, 'receive'))
        
        # Обновляем время узла, если он не промежуточный и не разрешены параллельные операции
        if not is_last_intermediate and not allow_parallel:
            self._update_time(last_node, receive_end, allow_parallel)
        
        # Обновляем информацию о времени передачи
        transfer_info['start_time'] = send_start if len(path) > 1 else receive_start
        transfer_info['end_time'] = receive_end
        
        return transfer_info


    def _update_time(self, node: int, new_time: float, allow_parallel: bool):
        """
        Обновляет текущее время узла с учетом параллелизма
        
        Параметры:
        ----------
        node : int
            ID узла
        new_time : float
            Новое время для проверки
        allow_parallel : bool
            Если True, разрешает параллельные операции (минимум)
            Если False, операции выполняются последовательно (максимум)
        """
        if not allow_parallel:
            # Последовательное выполнение: время не может уменьшаться
            if self.current_time[node] < new_time:
                self.current_time[node] = new_time
        else:
            # Параллельное выполнение: можно начинать операции раньше
            self.current_time[node] = min(self.current_time[node], new_time)

    def _fill_gaps(self, node: int):
        """
        Добавляет события "бездействия" между операциями в расписании
        
        Параметры:
        ----------
        node : int
            ID узла для обработки
        """
        events = sorted(self.schedule[node], key=lambda x: x[1])
        filled = []
        prev_end = 0.0
        for event in events:
            if event[1] > prev_end:
                # Добавление промежутка бездействия
                filled.append((
                    'Idle',
                    prev_end,
                    event[1],
                    'idle'
                ))
            filled.append(event)
            prev_end = event[2]
        self.schedule[node] = filled  # Обновление расписания с промежутками

    def get_shortest_path(self, source: int, target: int) -> List[int]:
        """
        Находит кратчайший путь между узлами в сети
        
        Параметры:
        ----------
        source : int
            Начальный узел
        target : int
            Конечный узел
        
        Возвращает:
        -----------
        path : list[int]
            Список узлов пути
        """
        return nx.shortest_path(self.net_graph.graph, source, target)

    def assign_tasks_to_nodes(self, distribution: List[int]) -> Dict[int, int]:
        """
        Создает отображение {task_id: node_id} на основе распределения
        
        Параметры:
        ----------
        distribution : list[int]
            Список, где индекс - ID задачи, значение - ID узла
        
        Возвращает:
        -----------
        assignments : dict
            Словарь соответствия задач и узлов
        """
        return {task_id: node_id for task_id, node_id in enumerate(distribution)}

    def get_total_execution_time(self):
        end_times = [max(tasks, key=lambda x: x[2])[2] for tasks in self.schedule.values()]
        return max(end_times)

    def create_gantt_chart(self):
        """
        Создает диаграмму Ганта с явным отображением операций отправки/получения:
        - Отдельные полосы для Send/Receive операций
        - Цветовая дифференциация всех типов операций
        - Детальная визуализация коммуникаций
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        plt.rcParams.update({
            'figure.figsize': (25.6, 14.4),
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.titlepad': 20,
            'axes.labelpad': 15,
            'legend.frameon': True,
            'legend.shadow': True
        })

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)

        # Цветовая схема с поддержкой Send/Receive
        model_colors = {
            1: {'task': '#FF6B6B', 'send': '#4ECDC4', 'receive': '#45B7D1', 'transfer': '#96CEB4'},
            2: {'task': '#FF8C00', 'send': '#66B2FF', 'receive': '#5CD6D6', 'transfer': '#9966FF'},
            3: {'task': '#2E8B57', 'send': '#66B2FF', 'receive': '#5CD6D6', 'transfer': '#FFD700'}
        }
        colors = model_colors.get(self.model_type, model_colors[1])

        # Параметры отображения
        node_height = 0.6
        transfer_thickness = 2.5
        total_time = self.get_total_execution_time()

        # Подготовка узлов
        all_nodes = list(self.net_graph.nodes.keys())
        
        if self.model_type == 2:
            # Для model_type=2 выделяем bus_node как отдельный
            bus_node = max(all_nodes)
            display_nodes = [n for n in all_nodes if n != bus_node] + [bus_node]
        elif self.model_type == 3:
            # Для model_type=3 разделяем оригинальные и промежуточные узлы
            original_count = len(self.original_net_graph.nodes)
            display_nodes = sorted(all_nodes, key=lambda x: (x >= original_count, x))
        else:
            display_nodes = all_nodes

        # Отрисовка узлов и операций
        y_ticks = []
        y_labels = []
        for idx, node_id in enumerate(display_nodes):
            y_pos = idx * 1.2
            y_ticks.append(y_pos)
            
            # Формирование подписи узла
            label = f"Node {node_id}"
            if self.model_type == 2 and node_id == max(display_nodes):
                label = f"BUS Node {node_id}"
            elif self.model_type == 3 and node_id >= len(self.original_net_graph.nodes):
                label = f"Intermediate Node {node_id}"
            y_labels.append(label)

            # Цвет фона узла
            bg_color = 'whitesmoke'
            if self.model_type == 2 and node_id == max(display_nodes):
                bg_color = '#F0F0F0'  # Светло-серый для bus_node
            elif self.model_type == 3 and node_id >= len(self.original_net_graph.nodes):
                bg_color = '#E0E0E0'  # Серый для промежуточных узлов

            # Фоновая полоса узла
            ax.barh(y_pos, total_time, height=node_height, left=0,
                    color=bg_color, edgecolor='gray', alpha=0.8)
            
            # Отрисовка операций
            if node_id in self.schedule:
                for entry in self.schedule[node_id]:
                    task, start, end, task_type = entry
                    duration = end - start

                    # Настройки цвета и стиля
                    if task_type == 'task':
                        color = colors['task']
                        bar_height = node_height * 0.8
                        alpha = 0.9
                        label_text = f'T{task}'
                        z_order = 1
                    elif task_type == 'send':
                        color = colors['send']
                        bar_height = node_height * 0.4
                        alpha = 0.7
                        label_text = 'SEND'
                        z_order = 3
                    elif task_type == 'receive':
                        color = colors['receive']
                        bar_height = node_height * 0.4
                        alpha = 0.7
                        label_text = 'RECV'
                        z_order = 3
                    elif task_type == 'transfer':
                        color = colors['transfer']
                        bar_height = node_height * 0.6
                        alpha = 0.6
                        label_text = 'TRANSFER'
                        z_order = 2
                    else:
                        continue

                    # Позиционирование для send/receive
                    y_offset = 0.3 * node_height
                    if task_type == 'send':
                        y_pos_adjusted = y_pos + y_offset
                    elif task_type == 'receive':
                        y_pos_adjusted = y_pos - y_offset
                    else:
                        y_pos_adjusted = y_pos

                    # Отрисовка полосы
                    ax.barh(y_pos_adjusted, duration, left=start,
                            height=bar_height, color=color, 
                            alpha=alpha, edgecolor='black', zorder=z_order)

                    # Добавление метки
                    if duration > 0.02 * total_time:
                        ax.text(start + duration/2, y_pos_adjusted, label_text,
                                ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.8),
                                zorder=z_order + 2)

        # Отрисовка передач данных
        for transfer in self.data_transfers:
            src, dst, start, end, task, succ = transfer
            duration = end - start

            try:
                y_src = display_nodes.index(src) * 1.2
                y_dst = display_nodes.index(dst) * 1.2
            except ValueError:
                continue

            # Стили для передачи
            color = colors['transfer']
            arrow_style = '-|>' if self.model_type != 2 else '->'
            connection_style = f"arc3,rad={0.3 if y_src != y_dst else 0}"

            if self.model_type == 2:
                # Для шины model_type=2 добавляем промежуточную точку
                bus_node = max(display_nodes)
                y_bus = display_nodes.index(bus_node) * 1.2
                mid_x = start + duration/2

                # Первый этап: отправитель → шина
                ax.annotate('', 
                        xy=(mid_x, y_bus), 
                        xytext=(start, y_src),
                        arrowprops=dict(arrowstyle=arrow_style, 
                                        color=color,
                                        linewidth=transfer_thickness,
                                        alpha=0.8,
                                        connectionstyle=connection_style))
                
                # Второй этап: шина → получатель
                ax.annotate('', 
                        xy=(end, y_dst), 
                        xytext=(mid_x, y_bus),
                        arrowprops=dict(arrowstyle=arrow_style, 
                                        color=color,
                                        linewidth=transfer_thickness,
                                        alpha=0.8,
                                        connectionstyle=connection_style))
                
                # Добавление меток для передачи
                ax.text(mid_x, (y_src + y_bus)/2, f"T{task} → Bus",
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8))
                ax.text(mid_x, (y_bus + y_dst)/2, f"Bus → T{succ}",
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8))
            else:
                # Прямая передача для других моделей
                ax.annotate('', 
                        xy=(end, y_dst), 
                        xytext=(start, y_src),
                        arrowprops=dict(arrowstyle=arrow_style, 
                                        color=color,
                                        linewidth=transfer_thickness,
                                        alpha=0.8,
                                        connectionstyle=connection_style))
                
                # Метка передачи
                mid_x = start + duration/2
                ax.text(mid_x, (y_src + y_dst)/2, f"T{task} → T{succ}",
                        ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.8))

        # Настройка осей и легенды
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Gantt Chart - Model {self.model_type} Schedule')
        ax.grid(True, axis='x', linestyle='--', alpha=0.7)
        ax.set_xlim(0, total_time * 1.05)

        # Легенда
        legend_elements = [
            mpatches.Patch(color=colors['task'], label='Task Execution'),
            mpatches.Patch(color=colors['send'], label='Send Operation'),
            mpatches.Patch(color=colors['receive'], label='Receive Operation'),
            mpatches.Patch(color=colors['transfer'], label='Data Transfer'),
            mpatches.Patch(color='#F0F0F0', label='Bus Node (Model 2)'),
            mpatches.Patch(color='#E0E0E0', label='Intermediate Node (Model 3)')
        ]
        ax.legend(handles=legend_elements, 
                loc='upper center', 
                bbox_to_anchor=(0.5, -0.15),
                ncol=3, 
                fancybox=True)

        plt.tight_layout()
        plt.show()

    def get_complete_analysis(self, distribution: list):
        """
        Полный анализ системы с учетом реальной нагрузки и временных характеристик
        """
        self.calculate_schedule(distribution)

        # Важно: создаем словарь для всех узлов, которые есть в расписании,
        # а не только для тех, что в original_net_graph
        analysis = {
            'nodes': {},
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
                'transfer_count': 0,
                'average_task_time': 0.0,
                'max_node_load': 0.0,
                'total_idle_time': 0.0,
                'average_transfer_time': 0.0
            }
        }
        
        # Инициализация данных для всех узлов, включая промежуточные
        for node_id in range(len(self.net_graph.nodes)):  # Используем метод nodes(), а не свойство nodes
            if isinstance(node_id, int):  # Проверка, что ID узла - целое число
                analysis['nodes'][node_id] = {
                    'performance': self.net_graph.get_node_performance(node_id),
                    'compute_load': 0.0,
                    'data_received': 0.0,
                    'data_sent': 0.0,
                    'working_time': 0.0,
                    'send_time': 0.0,
                    'receive_time': 0.0,
                    'idle_time': 0.0,
                    'utilization': 0.0
                }

        # 1. Анализ назначения задач и вычислительной нагрузки
        for task_id, node_id in self.node_assignments.items():
            task_complexity = self.task_graph.operations[task_id].get_task_complexity()
            analysis['nodes'][node_id]['compute_load'] += task_complexity
            analysis['statistics']['total_operations'] += task_complexity

        # 2. Анализ времени выполнения задач
        for node_id, tasks in self.schedule.items():
            # Убедимся, что узел существует в словаре анализа
            if node_id not in analysis['nodes']:
                analysis['nodes'][node_id] = {
                    'performance': self.net_graph.get_node_performance(node_id),
                    'compute_load': 0.0,
                    'data_received': 0.0,
                    'data_sent': 0.0,
                    'working_time': 0.0,
                    'send_time': 0.0,
                    'receive_time': 0.0,
                    'idle_time': 0.0,
                    'utilization': 0.0
                }
                
            for event in tasks:
                if event[3] == 'task':
                    task_id = event[0]
                    duration = event[2] - event[1]
                    analysis['tasks'][task_id]['execution_time'] = duration

        # 3. Анализ времени операций на узлах
        for node_id, tasks in self.schedule.items():
            # Проверяем, что узел существует в словаре анализа
            if node_id not in analysis['nodes']:
                continue
                
            node_stats = analysis['nodes'][node_id]
            for task_name, start, end, task_type in tasks:
                duration = end - start
                if task_type == 'task':
                    node_stats['working_time'] += duration
                elif task_type == 'send':
                    node_stats['send_time'] += duration
                elif task_type == 'receive':
                    node_stats['receive_time'] += duration
                elif task_type == 'idle':
                    node_stats['idle_time'] += duration

        # 4. Анализ передач данных
        for src, dst, start, end, task, successor in self.data_transfers:
            # Проверяем, что узлы существуют в словаре анализа
            if src not in analysis['nodes'] or dst not in analysis['nodes']:
                continue
                
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
                'end_time': end,
                'transfer_time': end - start
            }
            analysis['transfers'].append(transfer_info)

        # Рассчитываем общее время выполнения
        if self.schedule:
            max_end_times = []
            for node_id, tasks in self.schedule.items():
                if tasks:  # Проверяем, что список задач не пустой
                    max_end_time = max(end for _, _, end, _ in tasks)
                    max_end_times.append(max_end_time)
            
            if max_end_times:  # Проверяем, что список максимальных времен не пустой
                analysis['statistics']['total_time'] = max(max_end_times)

        # 5. Расчет дополнительных метрик для узлов
        for node_id, node_stats in analysis['nodes'].items():
            total_active_time = (node_stats['working_time'] +
                                node_stats['send_time'] +
                                node_stats['receive_time'])
            
            if analysis['statistics']['total_time'] > 0:
                node_stats['idle_time'] = analysis['statistics']['total_time'] - total_active_time
                node_stats['utilization'] = total_active_time / analysis['statistics']['total_time']
            else:
                node_stats['idle_time'] = 0
                node_stats['utilization'] = 0

        # 6. Финальные расчеты статистики
        analysis['statistics']['transfer_count'] = len(self.data_transfers)

        # Среднее время выполнения задач
        task_times = [task['execution_time'] for task in analysis['tasks'].values()]
        if task_times:
            analysis['statistics']['average_task_time'] = sum(task_times) / len(task_times)

        # Максимальная вычислительная нагрузка
        node_loads = [node['compute_load'] for node in analysis['nodes'].values()]
        if node_loads:
            analysis['statistics']['max_node_load'] = max(node_loads)

        # Общее время бездействия
        analysis['statistics']['total_idle_time'] = sum(
            node['idle_time'] for node in analysis['nodes'].values()
        )

        # Среднее время передачи данных
        if analysis['transfers']:
            total_transfer_time = sum(t['transfer_time'] for t in analysis['transfers'])
            analysis['statistics']['average_transfer_time'] = total_transfer_time / len(analysis['transfers'])

        return analysis


    def print_complete_analysis(self, distribution):
        """
        Вывод полного анализа системы с расширенными метриками вычислительной нагрузки
        """
        analysis = self.get_complete_analysis(distribution)
        
        print("\nАНАЛИЗ СИСТЕМЫ")
        print("=" * 50)
        
        # Сбор данных для общей статистики
        utilizations = []
        total_compute_load = 0.0
        total_performance_time = 0.0  # Сумма (производительность * рабочее время)
        bus_node = None
        
        # Информация об узлах
        print("\nХарактеристики и нагрузка узлов:")
        for node_id, node_info in analysis['nodes'].items():
            compute_load = node_info['compute_load']
            performance = node_info['performance']
            working_time = node_info['working_time']
            is_bus = node_info.get('is_bus', False)
            
            # Определяем магистраль для отдельного вывода
            if is_bus:
                bus_node = node_id
                continue  # Пропускаем вывод магистрали в общем списке
            
            print(f"\nУзел {node_id}:")
            print(f"  Производительность: {performance} оп/с")
            print(f"  Вычислительная нагрузка: {compute_load} операций")
            print(f"  Принято данных: {node_info['data_received']} байт")
            print(f"  Отправлено данных: {node_info['data_sent']} байт")
            print(f"  Время работы: {working_time:.2f} с")
            print(f"  Время на передачу: {node_info['send_time']:.2f} с")
            print(f"  Время на прием: {node_info['receive_time']:.2f} с")
            
        # Вывод информации о магистрали (если есть)
        if bus_node is not None:
            node_info = analysis['nodes'][bus_node]
            print(f"\nМагистраль (узел {bus_node}):")
            print(f"  Время работы: {node_info['working_time']:.2f} с")
            print(f"  Время передач: {node_info['send_time'] + node_info['receive_time']:.2f} с")

        # Общая статистика
        stats = analysis['statistics']
        print("\nОбщая статистика:")
        print(f"Общее время выполнения: {stats['total_time']:.2f} с")
        print(f"Общее количество операций: {stats['total_operations']}")
        print(f"Общий объем переданных данных: {stats['total_data_transferred']} байт")
        print(f"Количество передач данных: {stats['transfer_count']}")

        # Метрики загрузки узлов
        if utilizations:
            max_util = max(utilizations)
            min_util = min(utilizations)
            avg_util = sum(utilizations) / len(utilizations)
            print("\nМетрики загрузки вычислительных узлов:")
            print(f"  Максимальная загрузка: {max_util:.1f}%")
            print(f"  Минимальная загрузка: {min_util:.1f}%")
            print(f"  Средняя загрузка: {avg_util:.1f}%")
            if total_performance_time > 0:
                total_system_util = (total_compute_load / total_performance_time) * 100
                print(f"  Общая загрузка системы: {total_system_util:.1f}%")
            else:
                print("  Общая загрузка системы: Н/Д")
        else:
            print("\nМетрики загрузки: Н/Д (нет вычислительных узлов)")

        # Информация о магистрали/каналах
        if self.model_type == 2 and stats['bus_usage'] is not None:
            bus_usage_percent = (stats['bus_usage'] / stats['total_time']) * 100 if stats['total_time'] > 0 else 0.0
            print(f"\nИспользование магистрали: {bus_usage_percent:.1f}% времени")
        elif self.model_type == 3 and stats['channel_usage'] is not None:
            print("\nИспользование каналов связи:")
            for channel, time in stats['channel_usage'].items():
                print(f"  {channel}: {time:.2f} с")

        # Информация о задачах
        print("\nДетали выполнения задач:")
        for task_id, task_info in analysis['tasks'].items():
            print(f"\nЗадача {task_id}:")
            print(f"  Сложность: {task_info['complexity']} операций")
            print(f"  Назначена на узел: {task_info['assigned_node']}")
            print(f"  Время выполнения: {task_info['execution_time']:.2f} с")

        # Ключевые передачи данных
        if analysis['transfers']:
            print("\nКлючевые передачи данных:")
            for transfer in analysis['transfers'][:5]:  # Первые 5 передач
                print(f"\n  Передача от задачи {transfer['from_task']} к {transfer['to_task']}:")
                print(f"    Узлы: {transfer['from_node']} -> {transfer['to_node']}")
                print(f"    Объем: {transfer['data_volume']} байт")
                print(f"    Время: {transfer['start_time']:.2f} - {transfer['end_time']:.2f} с")
                if 'via_bus' in transfer:
                    print("    Тип: Через магистраль")
                elif 'channel_type' in transfer:
                    print(f"    Тип канала: {transfer['channel_type']}")

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
                 special_function_constraints: list = None,
                 model_type = 1,                          
                 intermediate_node_params = None,
                 intermediate_nodes_params = None,
                 bandwidth_factor=None   
                 ):
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
        self.model_type = model_type
        # Инициализация базовых параметров
        self.task_graph = task_graph
        self.t_lim = t_lim
        self.net_speed = net_speed
        self.original_net_graph = network_graph


        # Определение размерности вектора решения
        vector_length = task_graph.graph.number_of_nodes()

        if model_type == 1:
            self.network_graph = network_graph
            # Формирование ограничений на распределение
            bounds = self._create_constraints(bounds, vector_length)
        elif model_type == 2:
            default_constraints = {_:(0, max(self.original_net_graph.nodes)) for _ in range(vector_length)}
            self.network_graph = self._create_model2_net_graph(intermediate_node_params, bandwidth_factor)
            # Формирование ограничений на распределение
            bounds = self._create_constraints(default_constraints, vector_length)
        
        elif model_type == 3:
            default_constraints = {_:(0, max(self.original_net_graph.nodes)) for _ in range(vector_length)}
            self.network_graph = self._create_model3_net_graph(intermediate_nodes_params, bandwidth_factor)
            # Формирование ограничений на распределение
            bounds = self._create_constraints(default_constraints, vector_length)
        
        
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
        self.scheduler = TaskScheduler(task_graph, self.network_graph, model_type, self.original_net_graph)

    def _create_model2_net_graph(self, bus_params: dict, bandwidth_factor: float):
        """Создает граф с магистралью (model_type=2)"""
        bus_id = max(self.original_net_graph.nodes) + 1
        nodes = self.original_net_graph.nodes.copy()
        
        # Добавляем магистраль
        nodes[bus_id] = NetworkNode(
            id=bus_id,
            performance=bus_params.get('performance', 1000),
            cost=bus_params.get('cost', 0)
        )
        
        edges = []
        
        # Устанавливаем базовую пропускную способность для соединений с магистралью
        bus_bandwidth = bus_params.get('bandwidth', 1000)  # Предполагается, что bus_params содержит bandwidth
        
        # Создаем ребра от каждого исходного узла к магистрали
        for node_id, node in self.original_net_graph.nodes.items():
            edges.append((node_id, bus_id, bus_bandwidth * bandwidth_factor))

        return NetGraph(
            graph_type=3,
            nodes_params=nodes,
            edges=edges
        )

    def _create_model3_net_graph(self, nodes_params_list: list, bandwidth_factor: float):
        """Создает граф с промежуточными узлами (model_type=3)"""
        original_nodes = self.original_net_graph.nodes.copy()
        max_id = max(original_nodes.keys())
        
        intermediate_nodes = []
        edges = []
        i = 1
        # Создаем промежуточные узлы
        for idx, params in nodes_params_list.items():
            machine_id = max_id + i
            i += 1 
            original_nodes[machine_id] = params
            intermediate_nodes.append(machine_id)
            for node_id, node in self.original_net_graph.nodes.items():
                edges.extend([
                    (machine_id, node_id, params.get("bandwidth", 1000))
                ])
        
        return NetGraph(
            graph_type=3,
            nodes_params=original_nodes,
            edges=edges
        )

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