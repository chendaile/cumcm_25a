import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from numba import njit, cuda
import math


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


@cuda.jit
def gpu_calculate_diversity_kernel(velocities, distances, n_drones, population_size):
    """GPU核函数：计算种群多样性"""
    i = cuda.grid(1)

    if i < population_size * (population_size - 1) // 2:
        # 将一维索引转换为二维索引
        row = int((-1 + math.sqrt(1 + 8 * i)) / 2)
        col = i - row * (row + 1) // 2 + row + 1

        if col < population_size and row < population_size:
            distance = 0.0
            for drone_idx in range(n_drones):
                dist_x = velocities[row, drone_idx, 0] - \
                    velocities[col, drone_idx, 0]
                dist_y = velocities[row, drone_idx, 1] - \
                    velocities[col, drone_idx, 1]
                distance += dist_x * dist_x + dist_y * dist_y
            distances[i] = math.sqrt(distance)


@cuda.jit
def gpu_crossover_kernel(parent1_vel, parent2_vel, child_vel, alphas, noise, n_pop, n_drones):
    """GPU核函数：批量交叉操作"""
    i = cuda.grid(1)

    if i < n_pop:
        for drone_idx in range(n_drones):
            alpha = alphas[i]
            child_vel[i, drone_idx, 0] = (alpha * parent1_vel[i, drone_idx, 0] +
                                          (1 - alpha) * parent2_vel[i, drone_idx, 0] +
                                          noise[i, drone_idx, 0])
            child_vel[i, drone_idx, 1] = (alpha * parent1_vel[i, drone_idx, 1] +
                                          (1 - alpha) * parent2_vel[i, drone_idx, 1] +
                                          noise[i, drone_idx, 1])

            # 应用速度约束
            vx = child_vel[i, drone_idx, 0]
            vy = child_vel[i, drone_idx, 1]
            magnitude = math.sqrt(vx * vx + vy * vy)

            if magnitude < 70.0:
                scale = 70.0 / magnitude
                child_vel[i, drone_idx, 0] = vx * scale
                child_vel[i, drone_idx, 1] = vy * scale
            elif magnitude > 140.0:
                scale = 140.0 / magnitude
                child_vel[i, drone_idx, 0] = vx * scale
                child_vel[i, drone_idx, 1] = vy * scale


@cuda.jit
def gpu_mutate_kernel(velocities, noise, mutation_mask, n_pop, n_drones):
    """GPU核函数：批量变异操作"""
    i = cuda.grid(1)

    if i < n_pop:
        for drone_idx in range(n_drones):
            if mutation_mask[i, drone_idx]:
                vx = velocities[i, drone_idx, 0] + noise[i, drone_idx, 0]
                vy = velocities[i, drone_idx, 1] + noise[i, drone_idx, 1]

                # 应用速度约束
                magnitude = math.sqrt(vx * vx + vy * vy)
                if magnitude < 70.0:
                    scale = 70.0 / magnitude
                    velocities[i, drone_idx, 0] = vx * scale
                    velocities[i, drone_idx, 1] = vy * scale
                elif magnitude > 140.0:
                    scale = 140.0 / magnitude
                    velocities[i, drone_idx, 0] = vx * scale
                    velocities[i, drone_idx, 1] = vy * scale
                else:
                    velocities[i, drone_idx, 0] = vx
                    velocities[i, drone_idx, 1] = vy


@cuda.jit
def gpu_crossover_jammers_kernel(parent1_jammers, parent2_jammers, child_jammers,
                                 betas, noise_t, noise_delay, n_pop, n_jammers):
    """GPU核函数：批量干扰弹参数交叉"""
    i = cuda.grid(1)

    if i < n_pop:
        for j in range(n_jammers):
            beta = betas[i, j]
            father_t = (beta * parent1_jammers[i, j, 0] +
                        (1 - beta) * parent2_jammers[i, j, 0] +
                        noise_t[i, j])
            smoke_delay = (beta * parent1_jammers[i, j, 1] +
                           (1 - beta) * parent2_jammers[i, j, 1] +
                           noise_delay[i, j])

            # 应用约束
            child_jammers[i, j, 0] = max(0.0, min(5.0, father_t))
            child_jammers[i, j, 1] = max(0.0, min(5.0, smoke_delay))


@cuda.jit
def gpu_mutate_jammers_kernel(jammers, noise_t, noise_delay, mutation_mask, n_pop, n_jammers):
    """GPU核函数：批量干扰弹参数变异"""
    i = cuda.grid(1)

    if i < n_pop:
        for j in range(n_jammers):
            if mutation_mask[i, j]:
                father_t = jammers[i, j, 0] + noise_t[i, j]
                smoke_delay = jammers[i, j, 1] + noise_delay[i, j]

                jammers[i, j, 0] = max(0.0, min(5.0, father_t))
                jammers[i, j, 1] = max(0.0, min(5.0, smoke_delay))


def check_gpu_availability():
    """检查GPU是否可用"""
    try:
        cuda.detect()
        return True
    except Exception:
        return False


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
        self.use_gpu = check_gpu_availability()
        if self.use_gpu:
            print("GPU加速已启用 - 完整GPU版本")
        else:
            print("警告：GPU不可用，退回到CPU计算")

    def create_individual(self):
        with open('data-bin/ga_initial_params.json', 'r', encoding='utf-8') as f:
            params = json.load(f)

        individual = {}
        for drone_id in self.drone_ids:
            drone_params = params.get(drone_id, params['FY1'])
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
                jammers_array = np.array(jammers, dtype=np.float64)
                repaired_array = repair_jammers_timing(jammers_array)
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

    def gpu_batch_crossover(self, population, fitnesses):
        """GPU批量交叉操作"""
        if not self.use_gpu:
            return self.cpu_fallback_crossover(population)

        n_pop = len(population)
        n_drones = len(self.drone_ids)

        # 准备GPU数据
        parent1_vel = np.zeros((n_pop, n_drones, 2), dtype=np.float32)
        parent2_vel = np.zeros((n_pop, n_drones, 2), dtype=np.float32)
        parent1_jammers = np.zeros(
            (n_pop, self.n_jammers, 2), dtype=np.float32)
        parent2_jammers = np.zeros(
            (n_pop, self.n_jammers, 2), dtype=np.float32)

        # 选择父代并填充数据
        for i in range(n_pop):
            parent1 = self.tournament_selection(population, fitnesses)
            parent2 = self.tournament_selection(population, fitnesses)

            for j, drone_id in enumerate(self.drone_ids):
                parent1_vel[i, j, 0] = parent1[drone_id][0]
                parent1_vel[i, j, 1] = parent1[drone_id][1]
                parent2_vel[i, j, 0] = parent2[drone_id][0]
                parent2_vel[i, j, 1] = parent2[drone_id][1]

                for k in range(self.n_jammers):
                    if k < len(parent1[drone_id][2]):
                        parent1_jammers[i, k, 0] = parent1[drone_id][2][k][0]
                        parent1_jammers[i, k, 1] = parent1[drone_id][2][k][1]
                    if k < len(parent2[drone_id][2]):
                        parent2_jammers[i, k, 0] = parent2[drone_id][2][k][0]
                        parent2_jammers[i, k, 1] = parent2[drone_id][2][k][1]

        # GPU计算
        child_vel = np.zeros_like(parent1_vel)
        child_jammers = np.zeros_like(parent1_jammers)
        alphas = np.random.uniform(0.3, 0.7, n_pop).astype(np.float32)
        betas = np.random.uniform(
            0.2, 0.8, (n_pop, self.n_jammers)).astype(np.float32)
        noise_vel = np.random.normal(
            0, 5, (n_pop, n_drones, 2)).astype(np.float32)
        noise_t = np.random.normal(
            0, 0.1, (n_pop, self.n_jammers)).astype(np.float32)
        noise_delay = np.random.normal(
            0, 0.1, (n_pop, self.n_jammers)).astype(np.float32)

        # 传输到GPU并执行核函数
        d_parent1_vel = cuda.to_device(parent1_vel)
        d_parent2_vel = cuda.to_device(parent2_vel)
        d_child_vel = cuda.to_device(child_vel)
        d_alphas = cuda.to_device(alphas)
        d_noise_vel = cuda.to_device(noise_vel)

        threads_per_block = 256
        blocks_per_grid = (n_pop + threads_per_block - 1) // threads_per_block

        gpu_crossover_kernel[blocks_per_grid, threads_per_block](
            d_parent1_vel, d_parent2_vel, d_child_vel, d_alphas, d_noise_vel, n_pop, n_drones)

        # 干扰弹参数交叉
        d_parent1_jammers = cuda.to_device(parent1_jammers)
        d_parent2_jammers = cuda.to_device(parent2_jammers)
        d_child_jammers = cuda.to_device(child_jammers)
        d_betas = cuda.to_device(betas)
        d_noise_t = cuda.to_device(noise_t)
        d_noise_delay = cuda.to_device(noise_delay)

        gpu_crossover_jammers_kernel[blocks_per_grid, threads_per_block](
            d_parent1_jammers, d_parent2_jammers, d_child_jammers,
            d_betas, d_noise_t, d_noise_delay, n_pop, self.n_jammers)

        # 获取结果
        child_vel = d_child_vel.copy_to_host()
        child_jammers = d_child_jammers.copy_to_host()

        # 转换回individual格式
        new_population = []
        for i in range(n_pop):
            individual = {}
            for j, drone_id in enumerate(self.drone_ids):
                jammers = []
                for k in range(self.n_jammers):
                    jammers.append(
                        (child_jammers[i, k, 0], child_jammers[i, k, 1]))
                individual[drone_id] = [
                    child_vel[i, j, 0], child_vel[i, j, 1], jammers]
            new_population.append(self.repair_individual(individual))

        return new_population

    def gpu_batch_mutate(self, population, generation):
        """GPU批量变异操作"""
        if not self.use_gpu:
            return self.cpu_fallback_mutate(population, generation)

        base_mutation_rate = 0.25 if generation < self.generations // 2 else 0.15
        stagnation_boost = 1.0 + \
            (self.stagnation_counter / self.stagnation_threshold) * 2.0
        adaptive_rate = min(0.6, base_mutation_rate * stagnation_boost)
        intensity_factor = self.mutation_intensity * stagnation_boost

        n_pop = len(population)
        n_drones = len(self.drone_ids)

        # 准备数据
        velocities = np.zeros((n_pop, n_drones, 2), dtype=np.float32)
        jammers = np.zeros((n_pop, self.n_jammers, 2), dtype=np.float32)

        for i, individual in enumerate(population):
            for j, drone_id in enumerate(self.drone_ids):
                velocities[i, j, 0] = individual[drone_id][0]
                velocities[i, j, 1] = individual[drone_id][1]
                for k in range(min(self.n_jammers, len(individual[drone_id][2]))):
                    jammers[i, k, 0] = individual[drone_id][2][k][0]
                    jammers[i, k, 1] = individual[drone_id][2][k][1]

        # 生成变异参数
        base_noise = max(5, 30 - generation * 25 / self.generations)
        noise_scale = base_noise * intensity_factor
        if self.stagnation_counter > self.stagnation_threshold // 2:
            noise_scale *= 1.5

        mutation_mask_vel = np.random.random((n_pop, n_drones)) < adaptive_rate
        mutation_mask_jammers = np.random.random(
            (n_pop, self.n_jammers)) < (0.15 * intensity_factor)

        noise_vel = np.random.normal(
            0, noise_scale, (n_pop, n_drones, 2)).astype(np.float32)
        base_noise_t = max(0.1, 0.5 - generation * 0.4 /
                           self.generations) * intensity_factor
        base_noise_delay = max(0.1, 0.6 - generation *
                               0.5 / self.generations) * intensity_factor
        noise_t = np.random.normal(
            0, base_noise_t, (n_pop, self.n_jammers)).astype(np.float32)
        noise_delay = np.random.normal(
            0, base_noise_delay, (n_pop, self.n_jammers)).astype(np.float32)

        # GPU计算
        d_velocities = cuda.to_device(velocities)
        d_noise_vel = cuda.to_device(noise_vel)
        d_mutation_mask_vel = cuda.to_device(mutation_mask_vel)

        threads_per_block = 256
        blocks_per_grid = (n_pop + threads_per_block - 1) // threads_per_block

        gpu_mutate_kernel[blocks_per_grid, threads_per_block](
            d_velocities, d_noise_vel, d_mutation_mask_vel, n_pop, n_drones)

        d_jammers = cuda.to_device(jammers)
        d_noise_t = cuda.to_device(noise_t)
        d_noise_delay = cuda.to_device(noise_delay)
        d_mutation_mask_jammers = cuda.to_device(mutation_mask_jammers)

        gpu_mutate_jammers_kernel[blocks_per_grid, threads_per_block](
            d_jammers, d_noise_t, d_noise_delay, d_mutation_mask_jammers, n_pop, self.n_jammers)

        # 获取结果
        velocities = d_velocities.copy_to_host()
        jammers = d_jammers.copy_to_host()

        # 更新population
        for i, individual in enumerate(population):
            for j, drone_id in enumerate(self.drone_ids):
                individual[drone_id][0] = velocities[i, j, 0]
                individual[drone_id][1] = velocities[i, j, 1]
                for k in range(min(self.n_jammers, len(individual[drone_id][2]))):
                    individual[drone_id][2][k] = (
                        jammers[i, k, 0], jammers[i, k, 1])
            population[i] = self.repair_individual(individual)

        return population

    def calculate_diversity_gpu(self, population):
        """GPU加速的多样性计算"""
        if len(population) < 2 or not self.use_gpu:
            return 0.0

        n_pop = len(population)
        n_drones = len(self.drone_ids)
        velocities = np.zeros((n_pop, n_drones, 2), dtype=np.float64)

        for i, individual in enumerate(population):
            for j, drone_id in enumerate(self.drone_ids):
                velocities[i, j, 0] = individual[drone_id][0]
                velocities[i, j, 1] = individual[drone_id][1]

        n_pairs = n_pop * (n_pop - 1) // 2
        d_velocities = cuda.to_device(velocities)
        d_distances = cuda.device_array(n_pairs, dtype=np.float64)

        threads_per_block = 256
        blocks_per_grid = (n_pairs + threads_per_block -
                           1) // threads_per_block

        gpu_calculate_diversity_kernel[blocks_per_grid, threads_per_block](
            d_velocities, d_distances, n_drones, n_pop)

        distances = d_distances.copy_to_host()
        return np.mean(distances) if n_pairs > 0 else 0.0

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

    def cpu_fallback_crossover(self, population):
        """CPU回退版本的交叉操作"""
        print("GPU不可用，使用CPU交叉操作")
        return population

    def cpu_fallback_mutate(self, population, generation):
        """CPU回退版本的变异操作"""
        print("GPU不可用，使用CPU变异操作")
        return population

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

            diversity = self.calculate_diversity_gpu(population)
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

            # GPU批量操作
            remaining_size = self.population_size - len(new_population)
            if remaining_size > 0:
                batch_population = self.gpu_batch_crossover(
                    population, fitnesses)[:remaining_size]
                batch_population = self.gpu_batch_mutate(
                    batch_population, generation)
                new_population.extend(batch_population)

            population = new_population

        if plot_convergence:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.plot(range(1, len(best_fitness_history)+1), best_fitness_history,
                     'r-', linewidth=2, label='Best Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Coverage Duration (s)')
            plt.title('GPU-Enhanced GA Optimization Convergence')
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
            plt.savefig('tmp/gpu_genetic_algorithm_convergence.png',
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

        for drone_id, drone_data in result['drones'].items():
            self.global_system.reset_jammers(drone_id)
            self.global_system.update_drone_velocity(
                drone_id, [drone_data[0], drone_data[1], 0])
            for father_t, smoke_delay in drone_data[2]:
                self.global_system.add_jammers(drone_id, father_t, smoke_delay)

        cover_intervals = self.global_system.get_cover_intervals_all_jammers()

        with open(f'output/optimization_results_GPU_{self.Qname}.txt', 'a', encoding='utf-8') as f:
            f.write(f"GPU优化结果 - {timestamp}\n")
            f.write(f"覆盖时长: {result['duration']:.3f}秒\n")

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

        print(f"结果已保存到 output/optimization_results_GPU_{self.Qname}.txt")
