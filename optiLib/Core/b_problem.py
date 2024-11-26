# opti/problem.py

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
        base_objectives = [f(vector, self) for name, f in self.f_objective.items()]
        
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
