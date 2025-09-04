import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import json


class GeneticOptimizer:
    def __init__(self, global_system, drone_id, n_jammers, population_size, generations):
        self.global_system = global_system
        self.drone_id = drone_id
        self.n_jammers = n_jammers
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_fitness = 0

    def create_individual(self):
        with open('data-bin/ga_initial_params.json', 'r', encoding='utf-8') as f:
            params = json.load(f)
        velocity_x = params['velocity']['velocity_x']
        velocity_y = params['velocity']['velocity_y']
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)

        if velocity_magnitude < 70:
            scale = 70 / velocity_magnitude
            velocity_x *= scale
            velocity_y *= scale
        elif velocity_magnitude > 140:
            scale = 140 / velocity_magnitude
            velocity_x *= scale
            velocity_y *= scale

        jammers = []
        for _ in range(self.n_jammers):
            father_t = params['jammers']['father_t']
            smoke_delay = params['jammers']['smoke_delay']
            jammers.append((father_t, smoke_delay))
        return [velocity_x, velocity_y, jammers]

    def evaluate_individual(self, individual):
        self.global_system.reset_jammers(self.drone_id)
        self.global_system.update_drone_velocity(
            self.drone_id, [individual[0], individual[1], 0])

        for father_t, smoke_delay in individual[2]:
            self.global_system.add_jammers(
                self.drone_id, father_t, smoke_delay)
        return self.global_system.get_cover_seconds_all_jammers()

    def crossover(self, parent1, parent2):
        import random
        child = []
        alpha = random.uniform(0.3, 0.7)
        child.append(alpha * parent1[0] + (1 - alpha)
                     * parent2[0] + random.gauss(0, 5))
        child.append(alpha * parent1[1] + (1 - alpha)
                     * parent2[1] + random.gauss(0, 5))
        velocity_magnitude = np.sqrt(child[0]**2 + child[1]**2)
        if velocity_magnitude < 70:
            scale = 70 / velocity_magnitude
            child[0] *= scale
            child[1] *= scale
        elif velocity_magnitude > 140:
            scale = 140 / velocity_magnitude
            child[0] *= scale
            child[1] *= scale

        child_jammers = []
        for i in range(len(parent1[2])):
            if random.random() < 0.6:
                beta = random.uniform(0.2, 0.8)
                father_t = beta * parent1[2][i][0] + \
                    (1 - beta) * parent2[2][i][0]
                smoke_delay = beta * \
                    parent1[2][i][1] + (1 - beta) * parent2[2][i][1]
                child_jammers.append((father_t, smoke_delay))
            else:
                child_jammers.append(random.choice(
                    [parent1[2][i], parent2[2][i]]))
        child.append(child_jammers)
        return child

    def mutate(self, individual, generation):
        mutation_rate = 0.25 if generation < self.generations // 2 else 0.15

        if random.random() < mutation_rate:
            noise_scale = max(5, 30 - generation * 25 / self.generations)
            individual[0] += random.gauss(0, noise_scale)
            individual[1] += random.gauss(0, noise_scale)
            velocity_magnitude = np.sqrt(individual[0]**2 + individual[1]**2)
            if velocity_magnitude < 70:
                scale = 70 / velocity_magnitude
                individual[0] *= scale
                individual[1] *= scale
            elif velocity_magnitude > 140:
                scale = 140 / velocity_magnitude
                individual[0] *= scale
                individual[1] *= scale

        for i in range(len(individual[2])):
            if random.random() < 0.15:
                noise_t = max(0.1, 0.5 - generation * 0.4 / self.generations)
                noise_delay = max(0.1, 0.6 - generation *
                                  0.5 / self.generations)

                new_father_t = max(0.0, min(5.0,
                                            individual[2][i][0] + random.gauss(0, noise_t)))
                new_smoke_delay = max(0.0, min(5.0,
                                               individual[2][i][1] + random.gauss(0, noise_delay)))

                individual[2][i] = (new_father_t, new_smoke_delay)
        return individual

    def tournament_selection(self, population, fitnesses, tournament_size=3):
        tournament = random.choices(
            list(zip(population, fitnesses)), k=tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def optimize(self, plot_convergence=False):
        population = [self.create_individual()
                      for _ in range(self.population_size)]
        best_fitness_history = []

        for generation in range(self.generations):
            fitnesses = []
            for individual in population:
                fitness = self.evaluate_individual(individual)
                fitnesses.append(fitness)

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual.copy()
                    print(
                        f"Generation {generation+1}: New best {fitness:.3f}s")

            best_fitness_history.append(self.best_fitness)
            population_with_fitness = list(zip(population, fitnesses))
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)

            new_population = []
            elite_size = self.population_size // 4
            for i in range(elite_size):
                new_population.append(population_with_fitness[i][0])

            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, generation)
                new_population.append(child)

            population = new_population

        if plot_convergence:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, self.generations+1), best_fitness_history,
                     'r-', linewidth=2, label='Best Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Coverage Duration (s)')
            plt.title('Genetic Algorithm Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('tmp/genetic_algorithm_convergence.png',
                        dpi=800, bbox_inches='tight')
            plt.show()

        if self.best_individual:
            result = {
                'velocity': [self.best_individual[0], self.best_individual[1], 0],
                'jammers': self.best_individual[2],
                'duration': self.best_fitness
            }
            self.save_result_to_file(result)
            return result
        return None

    def save_result_to_file(self, result):
        if not os.path.exists('output'):
            os.makedirs('output')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('output/optimization_results.txt', 'a', encoding='utf-8') as f:
            f.write(f"优化结果 - {timestamp}\n")
            f.write(
                f"速度: [{result['velocity'][0]:.2f}, {result['velocity'][1]:.2f}, {result['velocity'][2]:.2f}]\n")
            f.write(f"覆盖时长: {result['duration']:.3f}秒\n")
            f.write(f"干扰弹参数:\n")
            for i, (father_t, smoke_delay) in enumerate(result['jammers']):
                f.write(
                    f"  干扰弹{i+1}: 发射时间={father_t:.2f}s, 烟雾延迟={smoke_delay:.2f}s\n")
            f.write("-" * 50 + "\n")

        print(f"结果已保存到 output/optimization_results.txt")
