# opti/problem.py

import numpy as np
import networkx as nx
from dataclasses import dataclass
import functools
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
import matplotlib.patches as mpatches


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
            return self.constrain_elements(np.random.choice(nods_to_select, size=tasks))
        else:
            if self.bounds is not None:
                lower_bounds, upper_bounds = self.bounds[:, 0], self.bounds[:, 1]
                return self.constrain_elements(np.random.uniform(lower_bounds, upper_bounds, self.vector_length).astype(self.dtype))
            else:
                return self.constrain_elements(np.zeros(self.vector_length, dtype=self.dtype))

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
    def __init__(self, task_graph: TaskGraph, net_graph: NetGraph):
        self.task_graph = task_graph
        self.net_graph = net_graph

    def assign_tasks_to_nodes(self, distribution):
        return {task: distribution[task] for task in range(len(distribution))}

    def shortest_path(self, start, end):
        return nx.shortest_path(self.net_graph.graph, start, end)

    def calculate_schedule(self, distribution: list):
        self.node_assignments = self.assign_tasks_to_nodes(distribution)
        self.schedule = defaultdict(list)
        self.data_transfers = []

        current_time = defaultdict(float)

        for task in nx.topological_sort(self.task_graph.graph):
            node = self.node_assignments[task]
            start_time = current_time[node]
            duration = self.task_graph.operations[task].w / self.net_graph.nodes[node].p
            end_time = start_time + duration
            self.schedule[node].append((task, start_time, end_time, 'task'))
            current_time[node] = end_time
            
            for successor in self.task_graph.graph.successors(task):
                successor_node = self.node_assignments[successor]
                if successor_node != node:
                    data_volume = self.task_graph.graph[task][successor]['weight']
                    path = self.shortest_path(node, successor_node)
                    
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
                            network_transfer_time = data_volume / self.net_graph.net_speed
                            transfer_start = max(current_time[current_node], current_time[next_node])
                            transfer_end = transfer_start + network_transfer_time
                            self.data_transfers.append((current_node, next_node, transfer_start, transfer_end, task, successor))
                            current_time[current_node] = transfer_end
                            current_time[next_node] = transfer_end


    def create_gantt_chart(self):
        fig, ax = plt.subplots(figsize=(15, 8))

        colors = plt.cm.Set3(np.linspace(0, 1, len(self.task_graph.graph.nodes())))
        send_color = 'lightblue'
        receive_color = 'lightgreen'
        transfer_color = 'lightgrey'
        idle_color = 'white'
        y_ticks = []
        y_labels = []

        total_time = self.get_total_execution_time()

        for i, node in enumerate(range(len(self.net_graph.nodes))):
            y_ticks.append(i)
            y_labels.append(f"Node {node}")
            
            # Добавляем фоновую полосу для всего времени выполнения
            ax.barh(i, total_time, left=0, height=0.5, align='center', color=idle_color, alpha=0.3)

            if node in self.schedule:
                for task, start, end, task_type in self.schedule[node]:
                    if task_type == 'task':
                        ax.barh(i, end - start, left=start, height=0.5, align='center', color=colors[int(task)], alpha=0.8)
                        ax.text((start + end) / 2, i, f'Task {task}', ha='center', va='center')
                    elif task_type == 'send':
                        ax.barh(i, end - start, left=start, height=0.3, align='center', color=send_color, alpha=0.5)
                        ax.text((start + end) / 2, i, task, ha='center', va='center', fontsize=8)
                    elif task_type == 'receive':
                        ax.barh(i, end - start, left=start, height=0.3, align='center', color=receive_color, alpha=0.5)
                        ax.text((start + end) / 2, i, task, ha='center', va='center', fontsize=8)

        for src, dst, start, end, task, successor in self.data_transfers:
            ax.barh((src + dst) / 2, end - start, left=start, height=0.1, align='center', color=transfer_color, alpha=0.5)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Time')
        ax.set_title('Gantt Chart of Task Execution and Data Transfer on Network Nodes')

        # Создаем легенду
        legend_elements = [
            mpatches.Patch(color=colors[0], alpha=0.8, label='Task Execution'),
            mpatches.Patch(color=send_color, alpha=0.5, label='Data Send'),
            mpatches.Patch(color=receive_color, alpha=0.5, label='Data Receive'),
            mpatches.Patch(color=transfer_color, alpha=0.5, label='Data Transfer'),
            mpatches.Patch(color=idle_color, alpha=0.3, label='Idle Time')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        plt.show()

    def print_info(self):
        print("Node Power:")
        for i, node in enumerate(self.net_graph.nodes):
            print(f"Node {i}: power {node.p}")

        print("\nTask Workload:")
        for i, operation in enumerate(self.task_graph.operations):
            print(f"Task {i}: workload {operation.w}")

        print("\nTask Assignment to Nodes:")
        for task, node in self.node_assignments.items():
            print(f"Task {task} assigned to node {node}")

        print("\nData Transfers:")
        for src, dst, start, end, task, successor in self.data_transfers:
            print(f"From task {task} to task {successor}: node {src} -> node {dst}, time: {start:.2f} - {end:.2f}")

    def get_total_execution_time(self):
        end_times = [max(tasks, key=lambda x: x[2])[2] for tasks in self.schedule.values()]
        return max(end_times)

    def get_node_send_times(self):
        node_send_times = defaultdict(float)
        for node, tasks in self.schedule.items():
            for task, start, end, task_type in tasks:
                if task_type == 'send':
                    node_send_times[node] += end - start
        return dict(node_send_times)

    def get_transfer_count(self):
        return len(self.data_transfers)

    def print_extended_info(self):
        print("\nРасширенная информация:")
        print(f"Общее время выполнения комплекса задач: {self.get_total_execution_time():.2f}")
        
        print("\nВремя, затраченное каждым узлом на отправку данных:")
        for node, time in self.get_node_send_times().items():
            print(f"Узел {node}: {time:.2f}")
        
        print(f"\nОбщее количество пересылок между задачами: {self.get_transfer_count()}")

    
    def print_extended_info(self):
        print("\nРасширенная информация:")
        print(f"Общее время выполнения комплекса задач: {self.get_total_execution_time():.2f}")
        
        print("\nВремя работы каждого узла:")
        for node, time in self.get_node_working_times().items():
            print(f"Узел {node}: {time:.2f}")

        print("\nВремя, затраченное каждым узлом на отправку данных:")
        for node, time in self.get_node_send_times().items():
            print(f"Узел {node}: {time:.2f}")
        
        print("\nВремя, затраченное каждым узлом на прием данных:")
        for node, time in self.get_node_receive_times().items():
            print(f"Узел {node}: {time:.2f}")
        
        print(f"\nОбщее количество пересылок между задачами: {self.get_transfer_count()}")

    def get_node_receive_times(self):
        node_receive_times = defaultdict(float)
        for node, tasks in self.schedule.items():
            for task, start, end, task_type in tasks:
                if task_type == 'receive':
                    node_receive_times[node] += end - start
        return dict(node_receive_times)
    
    def get_node_working_times(self):
        node_working_times = defaultdict(float)
        for node, tasks in self.schedule.items():
            for task, start, end, task_type in tasks:
                node_working_times[node] += end - start
        return dict(node_working_times)

class NetworkOptimizationProblem(OptimizationProblem):
    def __init__(self, network_graph, task_graph, f_objective, f_constraints=None, bounds=None, dtype = int, t_lim = 5, net_speed = 1000, name="NETproblem_1"):
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
        self.scheduler = TaskScheduler(task_graph, network_graph)