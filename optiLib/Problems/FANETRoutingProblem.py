from ..Core import OptimizationProblem


class FANETRoutingProblem(OptimizationProblem):
    """
    Класс, представляющий задачу оптимизации маршрутизации в сети FANET.
    
    Этот класс обрабатывает оптимизацию маршрутизации в летающих ad-hoc сетях (FANET),
    учитывая множественные цели, такие как энергопотребление и надежность соединения.
    
    Атрибуты:
        adj_matrix (np.ndarray): Матрица смежности, представляющая вероятности соединения
        coordinates (dict): Словарь, содержащий 3D координаты каждого узла
        source (int): Индекс исходного узла
        destination (int): Индекс целевого узла
        n_nodes (int): Общее количество узлов в сети
        
    Пример:
        >>> problem = FANETRoutingProblem(adj_matrix, coordinates, 0, 9)
        >>> solution = problem.generate_random_solution()
        >>> energy = problem.energy_consumption(solution, problem)
    """
    
    def __init__(self, adjacency_matrix, node_coordinates, source, destination, f_objective):
        """
        Инициализация задачи маршрутизации FANET.
        
        Аргументы:
            adjacency_matrix (np.ndarray): Матрица вероятностей соединения между узлами
            node_coordinates (dict): Словарь с координатами узлов {id_узла: [x, y, z]}
            source (int): Индекс начального узла
            destination (int): Индекс конечного узла
        
        Исключения:
            ValueError: Если размерности матрицы не соответствуют количеству координат
        """
        # Базовые параметры
        self.adj_matrix = adjacency_matrix
        self.coordinates = node_coordinates
        self.source = source
        self.destination = destination
        self.n_nodes = len(adjacency_matrix)
        
        
        # Создаем ограничения
        f_constraints = [
            self.path_constraint,
            self.connectivity_constraint
        ]
        
        # Границы для решения (путь может проходить через узлы от 0 до n_nodes-1)
        bounds = np.array([[0, self.n_nodes-1] for _ in range(self.n_nodes)])
        
        super().__init__(
            f_objective=f_objective,
            f_constraints=f_constraints,
            bounds=bounds,
            dtype=int,
            len=self.n_nodes
        )

    def path_constraint(self, solution, problem):
        """
        Проверка, что маршрут начинается в исходном узле и достигает целевого.
        
        Аргументы:
            solution (np.ndarray): Массив, представляющий путь через узлы
            problem: Экземпляр задачи
            
        Возвращает:
            bool: True если путь корректен, False в противном случае
        """
        # Проверяем, начинается ли путь с source
        if solution[0] != self.source:
            return False
        
        # Ищем destination в пути
        dest_found = False
        for i, node in enumerate(solution):
            if node == self.destination:
                dest_found = True
                break
        return dest_found

    def connectivity_constraint(self, solution, problem):
        """
        Проверка связности всех последовательных узлов в маршруте.
        
        Аргументы:
            solution (np.ndarray): Массив, представляющий путь через узлы
            problem: Экземпляр задачи
            
        Возвращает:
            bool: True если путь связен, False в противном случае
        """
        for i in range(len(solution)-1):
            if solution[i+1] == -1:
                break
            # Проверяем наличие связи между последовательными узлами
            if self.adj_matrix[solution[i]][solution[i+1]] == 0:
                return False
        return True