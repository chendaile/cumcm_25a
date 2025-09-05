import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from numba import njit


@njit
def repair_jammers_timing(jammers_times, min_interval=1.0):
    """加速的干扰弹时间修复函数"""
    if len(jammers_times) <= 1:
        return jammers_times

    # 按发射时间排序
    sorted_indices = np.argsort(jammers_times[:, 0])
    sorted_times = jammers_times[sorted_indices]

    # 调整间隔
    for i in range(1, len(sorted_times)):
        if sorted_times[i, 0] < sorted_times[i-1, 0] + min_interval:
            sorted_times[i, 0] = sorted_times[i-1, 0] + min_interval

    return sorted_times


@njit
def apply_velocity_constraints(vx, vy, min_speed=70.0, max_speed=140.0):
    """加速的速度约束函数"""
    magnitude = np.sqrt(vx**2 + vy**2)
    if magnitude < min_speed:
        scale = min_speed / magnitude
        return vx * scale, vy * scale
    elif magnitude > max_speed:
        scale = max_speed / magnitude
        return vx * scale, vy * scale
    return vx, vy


class GeneticOptimizer:
    def __init__(self, global_system, drone_ids, n_jammers, population_size, generations, Qname):
        self.Qname = Qname
        self.global_system = global_system
        self.drone_ids = [drone_ids] if isinstance(
            drone_ids, str) else drone_ids
        self.n_jammers = n_jammers
        self.population_size = population_size
        self.generations = generations
        self.best_individual = None
        self.best_fitness = 0

    def create_individual(self):
        with open('data-bin/ga_initial_params.json', 'r', encoding='utf-8') as f:
            params = json.load(f)

        individual = {}
        for drone_id in self.drone_ids:
            drone_params = params.get(
                drone_id, params['FY1'])  # 如果没有特定参数，使用FY1作为默认
            velocity_x = drone_params['velocity']['velocity_x']
            velocity_y = drone_params['velocity']['velocity_y']

            velocity_x, velocity_y = apply_velocity_constraints(
                velocity_x, velocity_y)

            jammers = []
            for _ in range(self.n_jammers):
                father_t = drone_params['jammers']['father_t']
                smoke_delay = drone_params['jammers']['smoke_delay']
                father_t = max(0.0, father_t)
                smoke_delay = max(0.0, smoke_delay)
                jammers.append((father_t, smoke_delay))

            individual[drone_id] = [velocity_x, velocity_y, jammers]

        return self.repair_individual(individual)

    def repair_individual(self, individual):
        """确保同一无人机的干扰弹发射时间至少间隔1秒"""
        for drone_id in individual:
            jammers = individual[drone_id][2]
            if len(jammers) > 1:
                # 转换为numpy数组加速处理
                jammers_array = np.array(jammers, dtype=np.float64)
                repaired_array = repair_jammers_timing(jammers_array)
                # 转换回tuple列表
                individual[drone_id][2] = [(t[0], t[1])
                                           for t in repaired_array]
        return individual

    def evaluate_individual(self, individual):
        for drone_id in self.drone_ids:
            self.global_system.reset_jammers(drone_id)
            drone_data = individual[drone_id]
            self.global_system.update_drone_velocity(
                drone_id, [drone_data[0], drone_data[1], 0])
            for father_t, smoke_delay in drone_data[2]:
                self.global_system.add_jammers(drone_id, father_t, smoke_delay)
        return self.global_system.get_cover_seconds_all_jammers()

    def crossover(self, parent1, parent2):
        child = {}
        for drone_id in self.drone_ids:
            alpha = random.uniform(0.3, 0.7)
            child_x = alpha * \
                parent1[drone_id][0] + (1 - alpha) * \
                parent2[drone_id][0] + random.gauss(0, 5)
            child_y = alpha * \
                parent1[drone_id][1] + (1 - alpha) * \
                parent2[drone_id][1] + random.gauss(0, 5)

            child_x, child_y = apply_velocity_constraints(child_x, child_y)

            child_jammers = []
            for i in range(len(parent1[drone_id][2])):
                if random.random() < 0.6:
                    beta = random.uniform(0.2, 0.8)
                    father_t = beta * \
                        parent1[drone_id][2][i][0] + \
                        (1 - beta) * parent2[drone_id][2][i][0]
                    smoke_delay = beta * \
                        parent1[drone_id][2][i][1] + \
                        (1 - beta) * parent2[drone_id][2][i][1]
                    child_jammers.append((father_t, smoke_delay))
                else:
                    child_jammers.append(random.choice(
                        [parent1[drone_id][2][i], parent2[drone_id][2][i]]))
            child[drone_id] = [child_x, child_y, child_jammers]
        return self.repair_individual(child)

    def mutate(self, individual, generation):
        mutation_rate = 0.25 if generation < self.generations // 2 else 0.15

        for drone_id in self.drone_ids:
            if random.random() < mutation_rate:
                noise_scale = max(5, 30 - generation * 25 / self.generations)
                individual[drone_id][0] += random.gauss(0, noise_scale)
                individual[drone_id][1] += random.gauss(0, noise_scale)

                individual[drone_id][0], individual[drone_id][1] = apply_velocity_constraints(
                    individual[drone_id][0], individual[drone_id][1])

            for i in range(len(individual[drone_id][2])):
                if random.random() < 0.15:
                    noise_t = max(0.1, 0.5 - generation *
                                  0.4 / self.generations)
                    noise_delay = max(0.1, 0.6 - generation *
                                      0.5 / self.generations)
                    new_father_t = max(
                        0.0, min(5.0, individual[drone_id][2][i][0] + random.gauss(0, noise_t)))
                    new_smoke_delay = max(
                        0.0, min(5.0, individual[drone_id][2][i][1] + random.gauss(0, noise_delay)))
                    individual[drone_id][2][i] = (
                        new_father_t, new_smoke_delay)
        return self.repair_individual(individual)

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
                'drones': self.best_individual,
                'duration': self.best_fitness
            }

            self.save_result_to_file(result)
            return result
        return None

    def save_result_to_file(self, result):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f'output/optimization_results_{self.Qname}.txt', 'a', encoding='utf-8') as f:
            f.write(f"优化结果 - {timestamp}\n")
            f.write(f"覆盖时长: {result['duration']:.3f}秒\n")

            for drone_id, drone_data in result['drones'].items():
                f.write(f"{drone_id}:\n")
                f.write(
                    f"  速度: [{drone_data[0]:.2f}, {drone_data[1]:.2f}, 0.00]\n")
                f.write(f"  干扰弹参数:\n")
                for i, (father_t, smoke_delay) in enumerate(drone_data[2]):
                    f.write(
                        f"    干扰弹{i+1}: 发射时间={father_t:.2f}s, 烟雾延迟={smoke_delay:.2f}s\n")
            f.write("-" * 50 + "\n")

        print(f"结果已保存到 output/optimization_results.txt")
