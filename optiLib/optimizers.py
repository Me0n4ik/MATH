# opti/optimizers.py
import copy
import numpy as np
from tqdm import tqdm
import random

class Optimizer:
    def __init__(self, problem, track_history=True, **kwargs):
        """
        :param problem: Экземпляр OptimizationProblem или подкласс OptimizationProblem.
        :param track_history: Флаг, указывающий, нужно ли вести историю итераций.
        """
        self.problem = problem
        self.track_history = track_history
        self.history = [] if track_history else None

    def optimize(self):
        """Метод, который должен быть реализован в подклассе."""
        raise NotImplementedError("Метод optimize() должен быть реализован в дочернем классе.")

    def update_history(self, iteration, vector):
        """Обновляет историю, если она включена."""
        if self.track_history:
            self.history.append({'iteration': iteration, 'vector': copy.copy(vector)})


class RandomSearchOptimizer(Optimizer):
    def __init__(self, problem, num_iterations=100):
        super().__init__(problem)
        self.num_iterations = num_iterations

    def optimize(self):
        best_solution = None
        best_value = float('inf')

        for _ in tqdm(range(self.num_iterations), desc="Optimizing"):
            solution = self.problem.generate_random_solution()
            value = self.problem.evaluate(solution)
            if value < best_value:
                best_solution = solution
                best_value = value

        return best_solution, best_value

class Particle:
    def __init__(self, problem):
        self.position = problem.generate_random_solution()
        self.velocity = np.random.uniform(-1, 1, problem.vector_length)
        self.best_position = np.copy(self.position)
        self.best_value = float('inf')

    def update_distribution(self):
        values = self.position + self.velocity
        self.position = np.clip(values, np.array(self.problem.bounds)[:, 0], np.array(self.problem.bounds)[:, 1])

    def update_velocity(self, global_best_position, omega, phi_p, phi_g):
        rp, rg = np.random.random(), np.random.random()
        self.velocity = (omega * self.velocity +
                         phi_p * rp * (self.best_position.position - self.position) +
                         phi_g * rg * (global_best_position.position - self.position)) 


class ParticleSwarmOptimizer(Optimizer):
    def __init__(self, problem, num_particles=30, iterations=1_0000, inertia=0.5, cognitive=1.5, social=1.5):
        super().__init__(problem)
        self.num_particles = num_particles
        self.iterations = iterations
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def optimize(self):
        dimensions = self.problem.vector_length
        bounds = np.array(self.problem.bounds)
        particles = [Particle(self.problem) for _ in range(self.num_particles)]
        
        global_best_position = Particle(self.problem).position
        global_best_value = float('inf')

        for _ in tqdm(range(self.iterations), desc="Optimizing"):
            for particle in particles:
                value = self.problem.evaluate(particle.position)
                if value < particle.best_value:
                    particle.best_value = value
                    particle.best_position = np.copy(particle.position)

                if value < global_best_value:
                    global_best_value = value
                    global_best_position = np.copy(particle.position)

            for particle in particles:
                cognitive_component = self.cognitive * np.random.rand(dimensions) * (particle.best_position - particle.position)
                social_component = self.social * np.random.rand(dimensions) * (global_best_position - particle.position)
                particle.velocity = self.inertia * particle.velocity + cognitive_component + social_component
                particle.position += particle.velocity.astype(self.problem.dtype)
                particle.position = self.problem.constrain_elements(particle.position)

        return global_best_position, global_best_value

class GeneticAlgorithmOptimizer(Optimizer):
    def __init__(self, problem, population_size=50, generations=100, mutation_rate=0.01, crossover_rate=0.9):
        super().__init__(problem)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def optimize(self):
        dimensions = self.problem.vector_length
        bounds = np.array(self.problem.bounds)

        def create_individual():
            return np.random.uniform(bounds[:, 0], bounds[:, 1], dimensions)

        def mutate(individual):
            for i in range(dimensions):
                if random.random() < self.mutation_rate:
                    individual[i] = np.random.uniform(bounds[i, 0], bounds[i, 1])
            return individual

        def crossover(parent1, parent2):
            if random.random() < self.crossover_rate:
                point = random.randint(1, dimensions - 1)
                child1 = np.concatenate((parent1[:point], parent2[point:]))
                child2 = np.concatenate((parent2[:point], parent1[point:]))
                return child1, child2
            return parent1, parent2

        def select(population):
            fitness = np.array([self.problem.evaluate(ind) for ind in population])
            probabilities = (1.0 / (1.0 + fitness)) / np.sum(1.0 / (1.0 + fitness))
            indices = np.arange(self.population_size)
            selected_indices = np.random.choice(indices, size=self.population_size, replace=True, p=probabilities)
            return population[selected_indices]

        population = [create_individual() for _ in range(self.population_size)]

        best_individual = None
        best_value = float('inf')

        for _ in range(self.generations):
            population = select(population)
            next_generation = []

            for i in range(0, self.population_size, 2):
                parent1, parent2 = population[i], population[i + 1]
                child1, child2 = crossover(parent1, parent2)
                next_generation.extend([mutate(child1), mutate(child2)])
            
            population = np.array(next_generation)

            for individual in population:
                value = self.problem.evaluate(individual)
                if value < best_value:
                    best_value = value
                    best_individual = individual

        return best_individual, best_value
