from copy import copy
from pprint import pprint
import numpy as np
from tqdm import tqdm
import random
import pandas as pd
import os


class Optimizer:
    def __init__(self, problem, track_history=True, update_history_coef = 1000):
        self.problem = problem
        self.algo_name = None
        self.track_history = track_history
        self.history = [] if track_history else None
        self.update_history_coef = update_history_coef
        self.update_history_counter = 0
        self.first_solution = False

    def optimize(self):
        raise NotImplementedError("Метод optimize() должен быть реализован в дочернем классе.")

    def update_history(self, iteration, vector):
        self.update_history_counter += 1
        if self.track_history:
            if not self.first_solution:
                if self.problem.evaluate(vector) != np.inf:
                    self.first_solution = True
                    self.history.append({'iteration': iteration, 'Решение': self.update_history_counter, 'Алгоритм': self.algo_name, 'vector': vector.copy()})
            else:
                if self.update_history_counter % self.update_history_coef == 0:
                    self.history.append({'iteration': iteration, 'Алгоритм': self.algo_name,'Решение': self.update_history_counter, 'vector': vector.copy()})


    def save(self):
        transformed_data = [{**d, **self.problem.get_info(d['vector'])} for d in self.history]
        # Создание DataFrame из списка словарей
        df = pd.DataFrame(transformed_data)

        # Экспорт DataFrame в Excel файл
        df.to_excel(f'./data/{self.problem.name}{self.algo_name}.xlsx', index=False)

    def save(self, experiment_number):
        transformed_data = [{**d, 'Эксперимент': experiment_number, **self.problem.get_info(d['vector'])} for d in self.history]
        
        # Создание DataFrame из списка словарей
        new_df = pd.DataFrame(transformed_data)

        filename = f'./data/{self.problem.name}{self.algo_name}.xlsx'

        if os.path.exists(filename):
            # Если файл существует, читаем его и добавляем новые данные
            existing_df = pd.read_excel(filename)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Сортируем DataFrame по номеру эксперимента и итерации
        combined_df = combined_df.sort_values(['Эксперимент', 'iteration'])

        # Экспорт DataFrame в Excel файл
        combined_df.to_excel(filename, index=False)

        print(f"Данные сохранены в файл: {filename}")

    def relod_data(self):
        self.history = []
        self.update_history_counter = 0

class RandomSearchOptimizer(Optimizer):
    def __init__(self, problem, iterations=100, track_history=True, update_history_coef = 10):
        super().__init__(problem, track_history, update_history_coef)
        self.iterations = iterations
        self.algo_name = "RS"

    def optimize(self):
        best_solution = self.problem.generate_random_solution()
        best_value = self.problem.evaluate(best_solution)

        for _ in tqdm(range(self.iterations), desc="Optimizing"):
            solution = self.problem.generate_random_solution()
            value = self.problem.evaluate(solution)
            if value < best_value:
                best_solution = solution
                best_value = value
            self.update_history(_, best_solution)
            if self.first_solution: 
                return best_solution, best_value
            
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
    def __init__(self, problem, num_particles=30, iterations=10000, inertia=0.7, cognitive=0.7, social=0.7, track_history=True, update_history_coef = 100):
        super().__init__(problem, track_history, update_history_coef)
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
                self.update_history(_, particle.position)
                if self.first_solution: 
                    return global_best_position, global_best_value

            for particle in particles:
                particle.update_velocity(global_best_position, self.inertia, self.cognitive, self.social)
                particle.update_position()
            
        return global_best_position, global_best_value


class GeneticAlgorithm(Optimizer):
    def __init__(self, problem, population_size=100, generations=1000, crossover_rate=0.8, mutation_rate=0.1, track_history=True, update_history_coef = 100):
        super().__init__(problem, track_history, update_history_coef)
        self.algo_name = "GA"
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def optimize(self):
        population = [self.problem.generate_random_solution() for _ in range(self.population_size)]
        
        best_solution = None
        best_value = np.inf

        for solution in population:
            current_value = self.problem.evaluate(solution)
            if current_value < best_value:
                best_solution = solution
                best_value = current_value
            self.update_history(0, solution)

            if self.first_solution: 
                return best_solution, best_value

        for generation in tqdm(range(self.generations), desc="Оптимизация"):
            new_population = []
            
            for _ in range(self.population_size // 2):
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.choices(population, k=2)
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = random.choices(population, k=2)
                
                if random.random() < self.mutation_rate:
                    child1 = self.mutate(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population

            for solution in population:
                current_value = self.problem.evaluate(solution)
                if current_value < best_value:
                    best_solution = solution
                    best_value = current_value

                self.update_history(generation, current_value)
                if self.first_solution: 
                    return best_solution, best_value
            
        return best_solution, best_value

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
        return child1, child2

    def mutate(self, individual):
        mutation_point = random.randint(0, len(individual) - 1)
        individual[mutation_point] = self.problem.generate_random_solution()[mutation_point]
        return individual

class DirectedRandomSearchOptimizer(Optimizer):
    def __init__(self, problem, iterations=1000, step_size=0.1, num_directions=10, track_history=True, update_history_coef=10):
        super().__init__(problem, track_history, update_history_coef)
        self.algo_name = "DRS"
        self.iterations = iterations
        self.step_size = step_size
        self.num_directions = num_directions

    def optimize(self):
        best_solution = self.problem.generate_random_solution()
        best_value = self.problem.evaluate(best_solution)

        for iteration in tqdm(range(self.iterations), desc="Optimizing"):
            # Генерируем случайные направления
            directions = np.random.uniform(-1, 1, (self.num_directions, self.problem.vector_length))
            
            # Нормализуем направления
            directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

            improved = False
            for direction in directions:
                # Пробуем сделать шаг в текущем направлении
                new_solution = best_solution + self.step_size * direction
                new_solution = self.problem.constrain_elements(new_solution)
                new_value = self.problem.evaluate(new_solution)

                if new_value < best_value:
                    best_solution = new_solution
                    best_value = new_value
                    improved = True
                    break  # Выходим из цикла, так как нашли улучшение

            # Если не было улучшения, уменьшаем размер шага
            if not improved:
                self.step_size *= 0.95  # Можно настроить коэффициент уменьшения

            self.update_history(iteration, best_solution)
            if self.first_solution: 
                return best_solution, best_value

        return best_solution, best_value

