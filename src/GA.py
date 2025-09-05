import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from numba import njit, types
from numba.typed import Dict, List


@njit
def repair_jammers_timing(jammers_times, min_interval=1.0):
    if len(jammers_times) <= 1:
        return jammers_times
    sorted_indices = np.argsort(jammers_times[:, 0])
    sorted_times = jammers_times[sorted_indices]
    for i in range(1, len(sorted_times)):
        if sorted_times[i, 0] < sorted_times[i-1, 0] + min_interval:
            sorted_times[i, 0] = sorted_times[i-1, 0] + min_interval
    return sorted_times


@njit
def apply_velocity_constraints(vx, vy, min_speed=70.0, max_speed=140.0):
    magnitude = np.sqrt(vx**2 + vy**2)
    if magnitude < min_speed:
        scale = min_speed / magnitude
        return vx * scale, vy * scale
    elif magnitude > max_speed:
        scale = max_speed / magnitude
        return vx * scale, vy * scale
    return vx, vy


@njit
def calculate_diversity_fast(velocities, n_drones, population_size):
    """快速计算种群多样性"""
    if population_size < 2:
        return 0.0

    total_distance = 0.0
    count = 0

    for i in range(population_size):
        for j in range(i+1, population_size):
            distance = 0.0
            for drone_idx in range(n_drones):
                dist_x = (velocities[i, drone_idx, 0] -
                          velocities[j, drone_idx, 0]) ** 2
                dist_y = (velocities[i, drone_idx, 1] -
                          velocities[j, drone_idx, 1]) ** 2
                distance += dist_x + dist_y
            total_distance += distance ** 0.5
            count += 1

    return total_distance / count if count > 0 else 0.0


@njit
def crossover_velocities(parent1_vel, parent2_vel, alpha, noise_scale):
    """快速交叉速度参数"""
    child_x = alpha * \
        parent1_vel[0] + (1 - alpha) * parent2_vel[0] + \
        np.random.normal(0, noise_scale)
    child_y = alpha * \
        parent1_vel[1] + (1 - alpha) * parent2_vel[1] + \
        np.random.normal(0, noise_scale)
    return apply_velocity_constraints(child_x, child_y)


@njit
def crossover_jammer_params(parent1_jammer, parent2_jammer, beta):
    """快速交叉干扰弹参数"""
    father_t = beta * parent1_jammer[0] + (1 - beta) * parent2_jammer[0]
    smoke_delay = beta * parent1_jammer[1] + (1 - beta) * parent2_jammer[1]
    return (max(0.0, min(5.0, father_t)), max(0.0, min(5.0, smoke_delay)))


@njit
def mutate_velocity_fast(velocity, noise_scale):
    """快速变异速度参数"""
    new_vx = velocity[0] + np.random.normal(0, noise_scale)
    new_vy = velocity[1] + np.random.normal(0, noise_scale)
    return apply_velocity_constraints(new_vx, new_vy)


@njit
def mutate_jammer_fast(jammer_params, noise_t, noise_delay):
    """快速变异干扰弹参数"""
    new_father_t = max(
        0.0, min(5.0, jammer_params[0] + np.random.normal(0, noise_t)))
    new_smoke_delay = max(
        0.0, min(5.0, jammer_params[1] + np.random.normal(0, noise_delay)))
    return (new_father_t, new_smoke_delay)


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
        self.stagnation_counter = 0
        self.stagnation_threshold = max(20, generations // 15)
        self.mutation_intensity = 1.0

    def create_individual(self):
        with open('data-bin/ga_initial_params.json', 'r', encoding='utf-8') as f:
            params = json.load(f)

        individual = {}
        for drone_id in self.drone_ids:
            drone_params = params.get(
                drone_id, params['FY1'])
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
            # 使用njit加速的速度交叉
            parent1_vel = np.array(
                [parent1[drone_id][0], parent1[drone_id][1]])
            parent2_vel = np.array(
                [parent2[drone_id][0], parent2[drone_id][1]])
            child_x, child_y = crossover_velocities(
                parent1_vel, parent2_vel, alpha, 5.0)

            child_jammers = []
            for i in range(len(parent1[drone_id][2])):
                if random.random() < 0.6:
                    beta = random.uniform(0.2, 0.8)
                    # 使用njit加速的干扰弹参数交叉
                    parent1_jammer = np.array(parent1[drone_id][2][i])
                    parent2_jammer = np.array(parent2[drone_id][2][i])
                    child_jammer = crossover_jammer_params(
                        parent1_jammer, parent2_jammer, beta)
                    child_jammers.append(child_jammer)
                else:
                    child_jammers.append(random.choice(
                        [parent1[drone_id][2][i], parent2[drone_id][2][i]]))
            child[drone_id] = [child_x, child_y, child_jammers]
        return self.repair_individual(child)

    def adaptive_mutate(self, individual, generation):
        base_mutation_rate = 0.25 if generation < self.generations // 2 else 0.15
        stagnation_boost = 1.0 + \
            (self.stagnation_counter / self.stagnation_threshold) * 2.0
        adaptive_rate = min(0.6, base_mutation_rate * stagnation_boost)

        intensity_factor = self.mutation_intensity * stagnation_boost

        for drone_id in self.drone_ids:
            if random.random() < adaptive_rate:
                base_noise = max(5, 30 - generation * 25 / self.generations)
                noise_scale = base_noise * intensity_factor

                if self.stagnation_counter > self.stagnation_threshold // 2:
                    noise_scale *= 1.5

                # 使用njit加速的速度变异
                current_vel = np.array(
                    [individual[drone_id][0], individual[drone_id][1]])
                new_vx, new_vy = mutate_velocity_fast(current_vel, noise_scale)
                individual[drone_id][0] = new_vx
                individual[drone_id][1] = new_vy

            jammer_mutation_rate = 0.15 * intensity_factor
            for i in range(len(individual[drone_id][2])):
                if random.random() < jammer_mutation_rate:
                    base_noise_t = max(
                        0.1, 0.5 - generation * 0.4 / self.generations)
                    base_noise_delay = max(
                        0.1, 0.6 - generation * 0.5 / self.generations)

                    noise_t = base_noise_t * intensity_factor
                    noise_delay = base_noise_delay * intensity_factor

                    # 使用njit加速的干扰弹参数变异
                    current_jammer = np.array(individual[drone_id][2][i])
                    new_jammer = mutate_jammer_fast(
                        current_jammer, noise_t, noise_delay)
                    individual[drone_id][2][i] = new_jammer
        return self.repair_individual(individual)

    def tournament_selection(self, population, fitnesses, tournament_size=3):
        tournament = random.choices(
            list(zip(population, fitnesses)), k=tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def restart_population(self, keep_best=True):
        print(
            f"Population restart triggered after {self.stagnation_counter} stagnations")
        new_population = []

        if keep_best and self.best_individual:
            new_population.append(self.best_individual.copy())

        while len(new_population) < self.population_size:
            new_population.append(self.create_individual())

        self.stagnation_counter = 0
        self.mutation_intensity = min(2.0, self.mutation_intensity * 1.5)
        return new_population

    def calculate_diversity(self, population):
        """计算种群多样性，使用njit加速"""
        if len(population) < 2:
            return 0.0

        # 转换为numpy数组以利用njit加速
        n_pop = len(population)
        n_drones = len(self.drone_ids)
        velocities = np.zeros((n_pop, n_drones, 2), dtype=np.float64)

        for i, individual in enumerate(population):
            for j, drone_id in enumerate(self.drone_ids):
                velocities[i, j, 0] = individual[drone_id][0]
                velocities[i, j, 1] = individual[drone_id][1]

        return calculate_diversity_fast(velocities, n_drones, n_pop)

    def optimize(self, plot_convergence=False):
        population = [self.create_individual()
                      for _ in range(self.population_size)]
        best_fitness_history = []
        diversity_history = []
        previous_best = 0

        for generation in range(self.generations):
            fitnesses = []
            print(f"Generation {generation+1}")
            for individual in population:
                fitness = self.evaluate_individual(individual)
                fitnesses.append(fitness)

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_individual = individual.copy()
                    self.stagnation_counter = 0
                    print(
                        f"Generation {generation+1}: New best {fitness:.3f}s")

            if self.best_fitness <= previous_best + 1e-6:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            previous_best = self.best_fitness

            diversity = self.calculate_diversity(population)
            best_fitness_history.append(self.best_fitness)
            diversity_history.append(diversity)

            if self.stagnation_counter >= self.stagnation_threshold:
                population = self.restart_population()
                continue

            population_with_fitness = list(zip(population, fitnesses))
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)

            new_population = []
            elite_size = max(1, self.population_size // 6)

            for i in range(elite_size):
                new_population.append(population_with_fitness[i][0])

            diversity_boost_size = max(2, self.population_size // 10)
            for _ in range(diversity_boost_size):
                new_population.append(self.create_individual())

            while len(new_population) < self.population_size:
                tournament_size = 3 if diversity > 50 else 5
                parent1 = self.tournament_selection(
                    population, fitnesses, tournament_size)
                parent2 = self.tournament_selection(
                    population, fitnesses, tournament_size)
                child = self.crossover(parent1, parent2)
                child = self.adaptive_mutate(child, generation)
                new_population.append(child)

            population = new_population

        if plot_convergence:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(best_fitness_history)+1), best_fitness_history,
                     'r-', linewidth=2, label='Best Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Coverage Duration (s)')
            plt.title('Enhanced GA Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(diversity_history)+1), diversity_history,
                     'b-', linewidth=2, label='Population Diversity')
            plt.xlabel('Generation')
            plt.ylabel('Diversity')
            plt.title('Population Diversity Evolution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig('tmp/enhanced_genetic_algorithm_convergence.png',
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

        # 获取遮挡时间间隔
        for drone_id, drone_data in result['drones'].items():
            self.global_system.reset_jammers(drone_id)
            self.global_system.update_drone_velocity(
                drone_id, [drone_data[0], drone_data[1], 0])
            for father_t, smoke_delay in drone_data[2]:
                self.global_system.add_jammers(drone_id, father_t, smoke_delay)

        cover_intervals = self.global_system.get_cover_intervals_all_jammers()

        with open(f'output/optimization_results_{self.Qname}.txt', 'a', encoding='utf-8') as f:
            f.write(f"优化结果 - {timestamp}\n")
            f.write(f"覆盖时长: {result['duration']:.3f}秒\n")

            # 写入遮挡时间间隔
            f.write(f"遮挡时间间隔:\n")
            if cover_intervals:
                for i, (start, end) in enumerate(cover_intervals):
                    f.write(
                        f"  区间{i+1}: {start:.2f}s - {end:.2f}s (持续: {end-start:.2f}s)\n")
            else:
                f.write("  无有效遮挡时间间隔\n")

            for drone_id, drone_data in result['drones'].items():
                f.write(f"{drone_id}:\n")
                f.write(
                    f"  速度: [{drone_data[0]:.2f}, {drone_data[1]:.2f}, 0.00]\n")
                f.write(f"  干扰弹参数:\n")
                for i, (father_t, smoke_delay) in enumerate(drone_data[2]):
                    f.write(
                        f"    干扰弹{i+1}: 发射时间={father_t:.2f}s, 烟雾延迟={smoke_delay:.2f}s\n")
            f.write("-" * 50 + "\n")

        print(f"结果已保存到 output/optimization_results_{self.Qname}.txt")
