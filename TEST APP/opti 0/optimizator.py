import numpy as np
import logging
from tqdm import tqdm
import copy

from .main import Vector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Базовый класс для реализаций оптимизаций
class Optimizer:
    def __init__(self, problem, track_history=True):
        """
        :param problem: Экземпляр OptimizationProblem или подкласс OptimizationProblem.
        :param track_history: Флаг, указывающий, нужно ли вести историю итераций.
        """
        self.problem = problem
        self.track_history = track_history
        self.history = [] if track_history else None

    def optimize(self):
        """Метод, который должен быть реализован в подклассе."""
        raise NotImplementedError("Метод 'optimize' должен быть реализован в подклассе.")

    def update_history(self, iteration, vector):
        """Обновляет историю, если она включена."""
        if self.track_history:
            self.history.append({'iteration': iteration, 'vector': copy.copy(vector)})

class Particl:
    def __init__(self):
        super().__init__(values)
        self.velocity = np.random.uniform(-1, 1, self.length_constraint)
        self.best_position = copy.copy(self)
        self.best_score = float('inf')

    def update_distribution(self):
        self.values = Particl(self.values + self.velocity)

    def update_velocity(self, global_best_position, omega, phi_p, phi_g):
        rp, rg = np.random.random(), np.random.random()
        self.velocity = (omega * self.velocity +
                         phi_p * rp * (self.best_position.values - self.values) +
                         phi_g * rg * (global_best_position.values - self.values)) 

class PSO(Optimizer):
    def __init__(self, problem, iterations=100_000, num_particles = 100, omega = 0.7, phi_p = 0.7, phi_g = 0.7, track_history=True):
        super().__init__(problem, track_history)
        self.iterations = iterations
        self.num_particles = num_particles
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.global_best_position = Particl()
        self.global_best_score = self.problem.convolution_evaluate_objectives(self.global_best_position)

    def optimize(self):

        self.particles = [Particl() for _ in range(self.num_particles)]

        f_calls = len(self.particles)
        self.update_history(f_calls, self.global_best_position)

        for _ in tqdm(range(self.iterations), desc="Optimizing"):
            for particle in self.particles:
                if self.problem.check_constraints(particle):
                    score = self.problem.convolution_evaluate_objectives(particle)
                else:
                    score = - np.inf

                if score > particle.best_score:
                    particle.best_position = particle
                    particle.best_score = score

                if score > self.global_best_score:
                    self.global_best_position = particle
                    self.global_best_score = score

            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.omega, self.phi_p, self.phi_g)
                particle.update_distribution()
                f_calls += 1

            self.update_history(f_calls, self.global_best_position)
