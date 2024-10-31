# opti/problem.py

import numpy as np
import networkx as nx
from dataclasses import dataclass
import functools
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.patches as mpatches
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D


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
    
    f_constraints : list
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

    def __init__(self, f_objective, f_constraints=None, bounds=None, dtype = int, len = 10, name = "Problem 1", node_functions=None, 
                 function_constraints=None, special_function_constraints=None):
        """
        Инициализация задачи оптимизации 
        
        Параметры:
        -----------
        f_objective : list
            Список основных целевых функций для оптимизации
        
        f_constraints : list, optional
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
        self.f_constraints = f_constraints if f_constraints is not None else []
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
            if any(node in vector for node in nodes):
                for func, bounds in constraints.items():
                    value = func(vector, self)
                    if not (bounds[0] <= value <= bounds[1]):
                        return False
                        
        return True


    def get_info(self, vector=None):
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
            **{f.__name__: f(vector, self) for f in self.f_constraints},
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
        base_objectives = [f(vector, self) for name, f in self.f_objective.items()]
        
        # Вычисляем значения специальных функций для конкретных узлов
        node_specific_objectives = []
        if vector is not None:
            for nodes, func in self.node_functions.items():
                # Проверяем наличие узлов из массива в текущем векторе
                if any(node in vector for node in nodes):
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
        return np.prod(all_objectives)

    def expanded_constraints(self, vector=None):
        """
        Проверяет выполнение всех функций-ограничений для заданного вектора.
        
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
        return [c(vector, self) for c in self.f_constraints]

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
        return all(c(vector, self) for c in self.f_constraints)

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
    def __init__(self, martx, net_power = (100, 2500), net_power_arr = None, e0 = (0,70), emax = (70,100), net_speed = 1000) -> None:
        self.graph = nx.Graph(np.array(martx))
        self.net_speed = net_speed
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
            return self.net_graph.graph[node1][node2]['weight']  
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
            duration = self.task_graph.operations[task].w / self.net_graph.nodes[node].p
            end_time = start_time + duration

            # Добавление задачи в расписание
            self.schedule[node].append((task, start_time, end_time, 'task'))
            current_time[node] = end_time
            
            # Обработка передачи данных преемникам
            for successor in self.task_graph.graph.successors(task):
                successor_node = self.node_assignments[successor]
                
                # Если задачи на разных узлах - планируем передачу данных
                if successor_node != node:
                    data_volume = self.task_graph.graph[task][successor]['weight']
                    path = self.shortest_path(node, successor_node)
                    
                    # Обработка каждого узла в пути передачи
                    for i in range(len(path)):
                        current_node = path[i]
                        
                        # Прием данных (кроме начального узла)
                        if i > 0:
                            receive_time = data_volume / self.net_graph.nodes[current_node].p
                            receive_start = current_time[current_node]
                            receive_end = receive_start + receive_time
                            self.schedule[current_node].append((f"Receive T{task}->{successor}", receive_start, receive_end, 'receive'))
                            current_time[current_node] = receive_end

                        # Отправка данных (кроме конечного узла)
                        if i < len(path) - 1:
                            send_time = data_volume / self.net_graph.nodes[current_node].p
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

    def print_info(self):
        """Вывод основной информации о распределении задач и характеристиках узлов"""
        self._print_node_powers()
        self._print_task_workloads()
        self._print_task_assignments()
        self._print_data_transfers()

    def _print_node_powers(self):
        """Вывод информации о вычислительной мощности узлов"""
        print("\nNode Power:")
        for i, node in enumerate(self.net_graph.nodes):
            print(f"Node {i}: power {node.p}")

    def _print_task_workloads(self):
        """Вывод информации о вычислительной сложности задач"""
        print("\nTask Workload:")
        for i, operation in enumerate(self.task_graph.operations):
            print(f"Task {i}: workload {operation.w}")

    def _print_task_assignments(self):
        """Вывод информации о распределении задач по узлам"""
        print("\nTask Assignment to Nodes:")
        for task, node in self.node_assignments.items():
            print(f"Task {task} assigned to node {node}")

    def _print_data_transfers(self):
        """Вывод информации о передачах данных между узлами"""
        print("\nData Transfers:")
        for src, dst, start, end, task, successor in self.data_transfers:
            print(f"From task {task} to task {successor}: "
                f"node {src} -> node {dst}, time: {start:.2f} - {end:.2f}")

    def get_timing_statistics(self):
        """
        Получение статистики времени для всех типов операций на узлах
        
        Returns:
            dict: Словарь со статистикой времени для каждого узла
        """
        return {
            'total_time': self.get_total_execution_time(),
            'working_times': self.get_node_working_times(),
            'send_times': self.get_node_send_times(),
            'receive_times': self.get_node_receive_times(),
            'transfer_count': self.get_transfer_count()
        }

    def get_node_send_times(self):
        """Расчет времени отправки данных для каждого узла"""
        return self._calculate_node_times('send')

    def get_node_receive_times(self):
        """Расчет времени приема данных для каждого узла"""
        return self._calculate_node_times('receive')

    def get_node_working_times(self):
        """Расчет общего рабочего времени для каждого узла"""
        node_working_times = defaultdict(float)
        for node, tasks in self.schedule.items():
            node_working_times[node] = sum(end - start for _, start, end, _ in tasks)
        return dict(node_working_times)

    def _calculate_node_times(self, operation_type):
        """
        Вспомогательный метод для расчета времени операций определенного типа
        
        Args:
            operation_type (str): Тип операции ('send' или 'receive')
        """
        node_times = defaultdict(float)
        for node, tasks in self.schedule.items():
            node_times[node] = sum(
                end - start 
                for _, start, end, task_type in tasks 
                if task_type == operation_type
            )
        return dict(node_times)

    def get_transfer_count(self):
        """Получение общего количества передач данных"""
        return len(self.data_transfers)

    def print_extended_info(self):
        """Вывод расширенной информации о выполнении задач"""
        stats = self.get_timing_statistics()
        
        print("\nРасширенная информация:")
        print(f"Общее время выполнения комплекса задач: {stats['total_time']:.2f}")
        
        self._print_timing_info("Время работы каждого узла", stats['working_times'])
        self._print_timing_info("Время отправки данных", stats['send_times'])
        self._print_timing_info("Время приема данных", stats['receive_times'])
        
        print("\nИнформация о скоростях передачи данных между узлами:")
        for edge in self.net_graph.graph.edges():
            speed = self.get_edge_speed(edge[0], edge[1])
            print(f"Между узлами {edge[0]} и {edge[1]}: {speed:.2f}")
        
        print(f"\nОбщее количество пересылок между задачами: {stats['transfer_count']}")

    def _print_timing_info(self, title, timing_dict):
        """
        Вспомогательный метод для вывода временной информации
        
        Args:
            title (str): Заголовок для вывода
            timing_dict (dict): Словарь с временными данными
        """
        print(f"\n{title}:")
        for node, time in timing_dict.items():
            print(f"Узел {node}: {time:.2f}")

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
                 f_constraints: list = None,
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
            f_constraints: Функции-ограничения для распределения
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
        constraints = self._create_constraints(bounds, vector_length)
        
        # Инициализация родительского класса
        super().__init__(
            f_objective=f_objective,
            f_constraints=f_constraints,
            bounds=constraints,
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
