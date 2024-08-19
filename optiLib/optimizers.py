from copy import copy
from pprint import pprint
import numpy as np
from tqdm import tqdm
import random
import pandas as pd


class Optimizer:
    def __init__(self, problem, track_history=True):
        self.problem = problem
        self.algo_name = None
        self.track_history = track_history
        self.history = [] if track_history else None

    def optimize(self):
        raise NotImplementedError("Метод optimize() должен быть реализован в дочернем классе.")

    def update_history(self, iteration, vector):
        if self.track_history:
            self.history.append({'iteration': iteration,'Алгоритм':self.algo_name, 'vector': vector.copy()})

    def save(self):
        transformed_data = [{**d, **self.problem.get_info(d['vector'])} for d in self.history]
        # Создание DataFrame из списка словарей
        df = pd.DataFrame(transformed_data)

        # Экспорт DataFrame в Excel файл
        df.to_excel(f'./data/{self.algo_name}.xlsx', index=False)


class RandomSearchOptimizer(Optimizer):
    def __init__(self, problem, iterations=100):
        super().__init__(problem)
        self.iterations = iterations
        self.algo_name = "RS"

    def optimize(self):
        best_solution = None
        best_value = float('inf')

        for _ in tqdm(range(self.iterations), desc="Optimizing"):
            solution = self.problem.generate_random_solution()
            value = self.problem.evaluate(solution)
            if value < best_value:
                best_solution = solution
                best_value = value
            self.update_history(_, best_solution)
        return best_solution, best_value


class Particle:
    def __init__(self, problem):
        self.problem = problem
        self.position = problem.generate_random_solution()
        self.velocity = np.random.uniform(-1, 1, problem.vector_length)
        self.best_position = self.position.copy()
        self.best_value = float('inf')

    def update_velocity(self, global_best_position, omega, phi_p, phi_g):
        rp, rg = np.random.random(), np.random.random()
        self.velocity = (omega * self.velocity +
                         phi_p * rp * (self.best_position - self.position) +
                         phi_g * rg * (global_best_position - self.position)) 

    def update_position(self):
        self.position += self.velocity.astype(self.problem.dtype)
        self.position = self.problem.constrain_elements(self.position)

    def evaluate(self):
        value = self.problem.evaluate(self.position)
        if value < self.best_value:
            self.best_value = value
            self.best_position = self.position.copy()
        return value


class ParticleSwarmOptimizer(Optimizer):
    def __init__(self, problem, num_particles=30, iterations=10000, inertia=0.7, cognitive=0.7, social=0.7):
        super().__init__(problem)
        self.algo_name = "PSO"
        self.num_particles = num_particles
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def optimize(self):
        particles = [Particle(self.problem) for _ in range(self.num_particles)]
        global_best_position = min(particles, key=lambda p: p.best_value).best_position
        global_best_value = min(particles, key=lambda p: p.best_value).best_value

        for _ in tqdm(range(self.iterations), desc="Optimizing"):
            for particle in particles:
                value = particle.evaluate()
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = particle.best_position.copy()

            for particle in particles:
                particle.update_velocity(global_best_position, self.inertia, self.cognitive, self.social)
                particle.update_position()
            self.update_history(_, global_best_position)
        return global_best_position, global_best_value
    
class GreedyOptimizer(Optimizer):
    def __init__(self, problem):
        super().__init__(problem)
        self.algo_name = "Жадный"

    def optimize(self):
        global_best_position = np.zeros_like(self.problem.generate_random_solution())
        global_best_value = self.problem.evaluate(global_best_position)

        for task_id in tqdm(range(len(global_best_position)), desc="Optimizing"):
            for node_id in range(self.problem.network_graph.graph.number_of_nodes()):
                new_solution = copy(global_best_position)
                new_solution[task_id] = node_id
                score = self.problem.evaluate(new_solution)
                if score < global_best_value:
                    global_best_position = new_solution
                    global_best_value = score
                self.update_history(task_id+node_id, global_best_value)
        return global_best_position, global_best_value
