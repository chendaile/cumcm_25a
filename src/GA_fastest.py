import datetime
import numpy as np
import matplotlib.pyplot as plt
# import random  # 未使用
import json
from numba import njit, prange
# import openpyxl  # 未使用
from concurrent.futures import ProcessPoolExecutor
# from concurrent.futures import ThreadPoolExecutor  # 未使用
import multiprocessing as mp
import hashlib
import gc
import platform
import time


@njit
def apply_velocity_constraints_vectorized(velocities, min_speed=70.0, max_speed=140.0):
    n_pop, n_drones = velocities.shape[:2]
    for i in prange(n_pop):
        for j in range(n_drones):
            vx, vy = velocities[i, j, 0], velocities[i, j, 1]
            magnitude = np.sqrt(vx**2 + vy**2)
            if magnitude < min_speed:
                scale = min_speed / magnitude
                velocities[i, j, 0] *= scale
                velocities[i, j, 1] *= scale
            elif magnitude > max_speed:
                scale = max_speed / magnitude
                velocities[i, j, 0] *= scale
                velocities[i, j, 1] *= scale


@njit
def repair_jammers_timing_vectorized(jammers_batch, min_interval=1.0):
    n_pop, n_drones, n_jammers = jammers_batch.shape[:3]
    for i in prange(n_pop):
        for j in range(n_drones):
            jammers = jammers_batch[i, j]
            if n_jammers > 1:
                sorted_indices = np.argsort(jammers[:, 0])
                sorted_times = jammers[sorted_indices]
                for k in range(1, n_jammers):
                    if sorted_times[k, 0] < sorted_times[k-1, 0] + min_interval:
                        sorted_times[k, 0] = sorted_times[k -
                                                          1, 0] + min_interval
                jammers_batch[i, j] = sorted_times


@njit
def vectorized_crossover(parent1_vel, parent2_vel, parent1_jam, parent2_jam,
                         crossover_prob, alpha_vel, alpha_jam):
    n_drones = parent1_vel.shape[0]
    n_jammers = parent1_jam.shape[1]

    child_vel = np.empty_like(parent1_vel)
    child_jam = np.empty_like(parent1_jam)

    for i in prange(n_drones):
        if np.random.random() < crossover_prob:
            child_vel[i, 0] = alpha_vel * parent1_vel[i, 0] + \
                (1 - alpha_vel) * parent2_vel[i, 0]
            child_vel[i, 1] = alpha_vel * parent1_vel[i, 1] + \
                (1 - alpha_vel) * parent2_vel[i, 1]
        else:
            child_vel[i] = parent1_vel[i] if np.random.random(
            ) < 0.5 else parent2_vel[i]

        for j in range(n_jammers):
            if np.random.random() < crossover_prob:
                child_jam[i, j, 0] = alpha_jam * parent1_jam[i,
                                                             j, 0] + (1 - alpha_jam) * parent2_jam[i, j, 0]
                child_jam[i, j, 1] = alpha_jam * parent1_jam[i,
                                                             j, 1] + (1 - alpha_jam) * parent2_jam[i, j, 1]
            else:
                child_jam[i, j] = parent1_jam[i,
                                              j] if np.random.random() < 0.5 else parent2_jam[i, j]

            child_jam[i, j, 0] = max(0.0, min(25.0, child_jam[i, j, 0]))
            child_jam[i, j, 1] = max(0.0, min(25.0, child_jam[i, j, 1]))

    return child_vel, child_jam


@njit
def vectorized_mutation(velocities, jammers, mutation_rates, noise_scales):
    n_pop, n_drones = velocities.shape[:2]
    n_jammers = jammers.shape[2]

    for i in prange(n_pop):
        for j in range(n_drones):
            if np.random.random() < mutation_rates[j]:
                velocities[i, j, 0] += np.random.normal(0, noise_scales[j])
                velocities[i, j, 1] += np.random.normal(0, noise_scales[j])

            for k in range(n_jammers):
                if np.random.random() < mutation_rates[j] * 0.6:
                    jammers[i, j, k,
                            0] += np.random.normal(0, noise_scales[j] * 0.1)
                    jammers[i, j, k,
                            1] += np.random.normal(0, noise_scales[j] * 0.1)
                    jammers[i, j, k, 0] = max(
                        0.0, min(25.0, jammers[i, j, k, 0]))
                    jammers[i, j, k, 1] = max(
                        0.0, min(25.0, jammers[i, j, k, 1]))


@njit
def tournament_selection_vectorized(fitnesses, n_selections, tournament_size=3):
    n_pop = len(fitnesses)
    selections = np.empty(n_selections, dtype=np.int32)

    for i in prange(n_selections):
        best_idx = np.random.randint(0, n_pop)
        best_fitness = fitnesses[best_idx]

        for _ in range(tournament_size - 1):
            candidate_idx = np.random.randint(0, n_pop)
            if fitnesses[candidate_idx] > best_fitness:
                best_fitness = fitnesses[candidate_idx]
                best_idx = candidate_idx

        selections[i] = best_idx

    return selections


class GeneticOptimizer:
    def __init__(self, global_system, drone_ids, n_jammers,
                 population_size, generations, Qname,
                 targeted_missile_ids=['M1']):
        self.targeted_missile_ids = targeted_missile_ids
        self.Qname = Qname
        self.global_system = global_system
        self.drone_ids = [drone_ids] if isinstance(
            drone_ids, str) else drone_ids
        self.n_drones = len(self.drone_ids)
        self.n_jammers = n_jammers
        self.population_size = population_size
        self.generations = generations
        self.n_processes = min(mp.cpu_count(), 8)

        self.population_velocities = np.zeros(
            (population_size, self.n_drones, 2), dtype=np.float32)
        self.population_jammers = np.zeros(
            (population_size, self.n_drones, n_jammers, 2), dtype=np.float32)

        self.fitness_cache = {}
        self.cache_hits = 0
        self.evaluations = 0

        self.best_individual = None
        self.best_fitness = 0
        self.stagnation_counter = 0
        self.stagnation_threshold = max(15, generations // 20)
        self.mutation_intensity = 1.0

        print(
            f"Fastest GA version - {self.n_processes} processes + cache + vectorization")

    def _hash_individual(self, velocities, jammers):
        combined = np.concatenate([velocities.flatten(), jammers.flatten()])
        rounded = np.round(combined, 1).astype(np.float32)
        return hashlib.md5(rounded.tobytes()).hexdigest()

    def create_individual(self):
        with open('data-bin/ga_initial_params.json', 'r', encoding='utf-8') as f:
            params = json.load(f)

        individual = {}
        for drone_id in self.drone_ids:
            drone_params = params.get(drone_id, params['FY1'])
            velocity_x = drone_params['velocity']['velocity_x']
            velocity_y = drone_params['velocity']['velocity_y']

            magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
            if magnitude < 70:
                scale = 70 / magnitude
                velocity_x *= scale
                velocity_y *= scale
            elif magnitude > 140:
                scale = 140 / magnitude
                velocity_x *= scale
                velocity_y *= scale

            jammers = []
            jammer_configs = drone_params['jammers']
            for i in range(min(self.n_jammers, len(jammer_configs))):
                father_t = max(0.0, jammer_configs[i]['father_t'])
                smoke_delay = max(0.0, jammer_configs[i]['smoke_delay'])
                jammers.append((father_t, smoke_delay))

            while len(jammers) < self.n_jammers:
                jammers.append((np.random.uniform(0, 20),
                               np.random.uniform(0, 15)))

            individual[drone_id] = [velocity_x, velocity_y, jammers]
        return individual

    def initialize_population_vectorized(self):
        for i in range(self.population_size):
            individual = self.create_individual()

            for j, drone_id in enumerate(self.drone_ids):
                self.population_velocities[i, j, 0] = individual[drone_id][0]
                self.population_velocities[i, j, 1] = individual[drone_id][1]

                for k in range(self.n_jammers):
                    self.population_jammers[i, j, k,
                                            0] = individual[drone_id][2][k][0]
                    self.population_jammers[i, j, k,
                                            1] = individual[drone_id][2][k][1]

        apply_velocity_constraints_vectorized(self.population_velocities)
        repair_jammers_timing_vectorized(self.population_jammers)

    def evaluate_population_parallel(self):
        chunk_size = max(1, self.population_size // self.n_processes)
        fitnesses = np.zeros(self.population_size, dtype=np.float32)

        eval_tasks = []
        # cached_indices = []  # 未使用

        for i in range(self.population_size):
            hash_key = self._hash_individual(
                self.population_velocities[i], self.population_jammers[i])

            if hash_key in self.fitness_cache:
                fitnesses[i] = self.fitness_cache[hash_key]
                # cached_indices.append(i)  # 未使用
                self.cache_hits += 1
            else:
                eval_tasks.append(i)

        if eval_tasks:
            chunks = []
            for start in range(0, len(eval_tasks), chunk_size):
                end = min(start + chunk_size, len(eval_tasks))
                chunk_indices = eval_tasks[start:end]
                chunks.append(chunk_indices)

            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                chunk_results = list(executor.map(
                    self._evaluate_chunk, chunks))

            for chunk_indices, chunk_fitnesses in chunk_results:
                for idx, fitness in zip(chunk_indices, chunk_fitnesses):
                    fitnesses[idx] = fitness
                    hash_key = self._hash_individual(
                        self.population_velocities[idx], self.population_jammers[idx])
                    self.fitness_cache[hash_key] = fitness

        self.evaluations += len(eval_tasks)

        if len(self.fitness_cache) > 20000:
            keys = list(self.fitness_cache.keys())
            self.fitness_cache = {
                k: self.fitness_cache[k] for k in keys[-10000:]}
            gc.collect()

        return fitnesses

    def _evaluate_chunk(self, chunk_indices):
        fitnesses = []
        for idx in chunk_indices:
            individual_dict = self._convert_to_dict(idx)
            fitness = self._evaluate_individual_dict(individual_dict)
            fitnesses.append(fitness)
        return chunk_indices, fitnesses

    def _convert_to_dict(self, idx):
        individual = {}
        for j, drone_id in enumerate(self.drone_ids):
            vx = float(self.population_velocities[idx, j, 0])
            vy = float(self.population_velocities[idx, j, 1])
            jammers = []
            for k in range(self.n_jammers):
                father_t = float(self.population_jammers[idx, j, k, 0])
                smoke_delay = float(self.population_jammers[idx, j, k, 1])
                jammers.append((father_t, smoke_delay))
            individual[drone_id] = [vx, vy, jammers]
        return individual

    def _evaluate_individual_dict(self, individual):
        for drone_id in self.drone_ids:
            self.global_system.reset_jammers(drone_id)
            drone_data = individual[drone_id]
            self.global_system.update_drone_velocity(
                drone_id, [drone_data[0], drone_data[1], 0])
            for father_t, smoke_delay in drone_data[2]:
                self.global_system.add_jammers(drone_id, father_t, smoke_delay)

        return self.global_system.get_cover_duration(self.targeted_missile_ids)

    def evolve_generation_vectorized(self, generation):
        fitnesses = self.evaluate_population_parallel()

        best_idx = np.argmax(fitnesses)
        current_best = fitnesses[best_idx]
        avg_fitness = np.mean(fitnesses)

        if current_best > self.best_fitness:
            self.best_fitness = current_best
            self.best_individual = self._convert_to_dict(best_idx)
            self.stagnation_counter = 0
            print(f"Generation {generation+1}: New best {current_best:.3f}s")
        else:
            self.stagnation_counter += 1

        if self.stagnation_counter >= self.stagnation_threshold:
            self._restart_population()
            return avg_fitness, True  # 返回重启标志

        sorted_indices = np.argsort(fitnesses)[::-1]
        elite_size = max(2, self.population_size // 10)

        new_velocities = np.zeros_like(self.population_velocities)
        new_jammers = np.zeros_like(self.population_jammers)

        new_velocities[:elite_size] = self.population_velocities[sorted_indices[:elite_size]]
        new_jammers[:elite_size] = self.population_jammers[sorted_indices[:elite_size]]

        current_idx = elite_size

        while current_idx < self.population_size:
            n_offspring = min(self.population_size - current_idx,
                              (self.population_size - elite_size) // 2 * 2)

            parent_indices = tournament_selection_vectorized(
                fitnesses, n_offspring, tournament_size=3)

            for i in range(0, len(parent_indices), 2):
                if current_idx >= self.population_size:
                    break

                parent1_idx = parent_indices[i]
                parent2_idx = parent_indices[min(i+1, len(parent_indices)-1)]

                crossover_prob = 0.8
                alpha_vel = np.random.uniform(0.3, 0.7)
                alpha_jam = np.random.uniform(0.2, 0.8)

                child1_vel, child1_jam = vectorized_crossover(
                    self.population_velocities[parent1_idx],
                    self.population_velocities[parent2_idx],
                    self.population_jammers[parent1_idx],
                    self.population_jammers[parent2_idx],
                    crossover_prob, alpha_vel, alpha_jam
                )

                new_velocities[current_idx] = child1_vel
                new_jammers[current_idx] = child1_jam
                current_idx += 1

                if current_idx < self.population_size:
                    child2_vel, child2_jam = vectorized_crossover(
                        self.population_velocities[parent2_idx],
                        self.population_velocities[parent1_idx],
                        self.population_jammers[parent2_idx],
                        self.population_jammers[parent1_idx],
                        crossover_prob, alpha_jam, alpha_vel
                    )

                    new_velocities[current_idx] = child2_vel
                    new_jammers[current_idx] = child2_jam
                    current_idx += 1

        self.population_velocities = new_velocities
        self.population_jammers = new_jammers

        base_mutation_rate = 0.2 if generation < self.generations // 2 else 0.1
        stagnation_boost = 1.0 + self.stagnation_counter * 0.3
        mutation_rates = np.full(
            self.n_drones, base_mutation_rate * stagnation_boost * self.mutation_intensity)

        base_noise = max(8, 25 - generation * 15 / self.generations)
        noise_scales = np.full(self.n_drones, base_noise * stagnation_boost)

        vectorized_mutation(
            self.population_velocities[elite_size:],
            self.population_jammers[elite_size:],
            mutation_rates, noise_scales
        )

        apply_velocity_constraints_vectorized(self.population_velocities)
        repair_jammers_timing_vectorized(self.population_jammers)

        return avg_fitness, False  # 没有重启

    def _restart_population(self):
        print(
            f"Population restart after {self.stagnation_counter} stagnations")

        keep_size = max(1, self.population_size // 8)

        for i in range(keep_size, self.population_size):
            for j in range(self.n_drones):
                velocity_mag = np.random.uniform(70, 140)
                angle = np.random.uniform(0, 2 * np.pi)
                self.population_velocities[i, j,
                                           0] = velocity_mag * np.cos(angle)
                self.population_velocities[i, j,
                                           1] = velocity_mag * np.sin(angle)

                for k in range(self.n_jammers):
                    self.population_jammers[i, j, k,
                                            0] = np.random.uniform(0, 20)
                    self.population_jammers[i, j, k,
                                            1] = np.random.uniform(0, 15)

        apply_velocity_constraints_vectorized(self.population_velocities)
        repair_jammers_timing_vectorized(self.population_jammers)

        self.stagnation_counter = 0
        self.mutation_intensity = min(2.0, self.mutation_intensity * 1.3)

    def optimize(self, plot_convergence=False):
        optimization_start = time.time()
        self.initialize_population_vectorized()

        # 收集详细统计信息
        best_fitness_history = []
        avg_fitness_history = []
        cache_hit_ratios = []
        stagnation_history = []
        mutation_intensity_history = []
        evaluation_count_history = []
        improvement_generations = []
        restart_generations = []  # 记录重启发生的代数

        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")

            # 记录进化前状态
            old_best = self.best_fitness

            # 记录进化前的变异强度
            pre_evolution_mutation_intensity = self.mutation_intensity

            result = self.evolve_generation_vectorized(generation)
            if isinstance(result, tuple):
                avg_fitness, restarted = result
            else:
                avg_fitness, restarted = result, False

            # 记录统计信息
            best_fitness_history.append(self.best_fitness)
            if avg_fitness is not None:
                avg_fitness_history.append(avg_fitness)
            else:
                avg_fitness_history.append(self.best_fitness)  # 重启时使用最佳值

            # 如果发生了重启，记录重启代数和重启前的停滞计数
            if restarted:
                restart_generations.append(generation + 1)
                # 重启时记录达到阈值的停滞计数
                stagnation_history.append(self.stagnation_threshold)
            else:
                stagnation_history.append(self.stagnation_counter)

            # 记录更新后的变异强度（包含重启时的增加）
            mutation_intensity_history.append(self.mutation_intensity)
            evaluation_count_history.append(self.evaluations)

            # 记录改进的世代
            if self.best_fitness > old_best:
                improvement_generations.append(generation + 1)

            if self.evaluations + self.cache_hits > 0:
                hit_ratio = self.cache_hits / \
                    (self.evaluations + self.cache_hits)
                cache_hit_ratios.append(hit_ratio)
                if generation % 10 == 0:
                    print(f"Cache hit ratio: {hit_ratio:.3f}")

        # 高级可视化功能 - 详细分析优化过程
        if plot_convergence:
            # 设置美观的样式
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except OSError:
                try:
                    plt.style.use('seaborn-darkgrid')
                except OSError:
                    plt.style.use('default')
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Genetic Algorithm Optimization Analysis',
                         fontsize=16, fontweight='bold')

            # 子图1: 适应度收敛曲线
            ax1 = axes[0, 0]
            generations = list(range(1, len(best_fitness_history)+1))

            # 确保数组长度一致
            min_len = min(len(best_fitness_history), len(
                avg_fitness_history), len(generations))
            generations = generations[:min_len]
            best_vals = best_fitness_history[:min_len]
            avg_vals = avg_fitness_history[:min_len]

            ax1.plot(generations, best_vals, 'r-', linewidth=2.5,
                     label='Best Fitness', marker='o', markersize=2)
            ax1.plot(generations, avg_vals, 'b--', linewidth=1.5,
                     label='Average Fitness', alpha=0.7)

            # Fill between average and best fitness
            if len(generations) > 0:
                where_fill = np.array(avg_vals) <= np.array(best_vals)
                if np.any(where_fill):
                    ax1.fill_between(generations, avg_vals, best_vals,
                                     where=where_fill, alpha=0.2, color='green',
                                     label='Fitness Gap')

            # 标记改进点
            if improvement_generations:
                improvement_values = []
                for i in improvement_generations:
                    if 0 < i <= len(best_vals):
                        improvement_values.append(best_vals[i-1])
                if improvement_values:
                    valid_gens = [
                        g for g in improvement_generations if 0 < g <= len(best_vals)]
                    ax1.scatter(valid_gens, improvement_values,
                                color='gold', s=50, zorder=5, label='Improvement', marker='*')

            ax1.set_xlabel('Generation', fontsize=12)
            ax1.set_ylabel('Coverage Duration (s)', fontsize=12)
            ax1.set_title('Fitness Convergence Analysis',
                          fontsize=13, fontweight='bold')
            ax1.legend(loc=(0.65, 0.75))
            ax1.grid(True, alpha=0.3)

            # 子图2: 缓存效率分析
            ax2 = axes[0, 1]
            if cache_hit_ratios and len(cache_hit_ratios) > 0:
                cache_len = min(len(cache_hit_ratios), len(generations))
                if cache_len > 0:
                    gens_cache = generations[:cache_len]
                    ratios_pct = [r*100 for r in cache_hit_ratios[:cache_len]]

                    # 确保数组是真正的列表而不是0维数组
                    gens_cache = list(gens_cache) if hasattr(
                        gens_cache, '__iter__') else [gens_cache]
                    ratios_pct = list(ratios_pct) if hasattr(
                        ratios_pct, '__iter__') else [ratios_pct]

                    if len(gens_cache) > 1 and len(ratios_pct) > 1:  # 需要至少2个点才能绘图
                        ax2.plot(gens_cache, ratios_pct, 'g-',
                                 linewidth=2, marker='s', markersize=3)
                        try:
                            ax2.fill_between(
                                gens_cache, 0, ratios_pct, alpha=0.3, color='green')
                        except (IndexError, ValueError):
                            pass  # 如果fill_between失败就跳过
                        final_ratio = cache_hit_ratios[-1] * 100
                        ax2.axhline(y=final_ratio, color='red', linestyle='--',
                                    label=f'Final Hit Rate: {final_ratio:.1f}%')
                    elif len(gens_cache) >= 1 and len(ratios_pct) >= 1:
                        ax2.scatter(gens_cache, ratios_pct,
                                    color='green', s=50, marker='s')
                ax2.set_ylabel('Cache Hit Rate (%)', fontsize=12)
            ax2.set_xlabel('Generation', fontsize=12)
            ax2.set_title('Cache Efficiency Analysis',
                          fontsize=13, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 子图3: 停滞与重启分析
            ax3 = axes[0, 2]
            stag_len = min(len(stagnation_history), len(generations))
            gens_stag = generations[:stag_len]
            stag_vals = stagnation_history[:stag_len]

            if len(gens_stag) > 0:
                ax3.plot(gens_stag, stag_vals, 'orange', linewidth=2,
                         marker='d', markersize=3, label='Stagnation Count')
            ax3.axhline(y=self.stagnation_threshold, color='red', linestyle='--',
                        label=f'Restart Threshold: {self.stagnation_threshold}')

            # Mark restart points using recorded restart generations
            if restart_generations:
                # 标记在重启后的下一代，即下降后的拐点位置(y=0)
                restart_next_gen = [
                    gen + 1 for gen in restart_generations if gen + 1 <= len(generations)]
                if restart_next_gen:
                    ax3.scatter(restart_next_gen, [0] * len(restart_next_gen),
                                color='red', s=80, marker='X', zorder=5, label='Population Restart')

            ax3.set_xlabel('Generation', fontsize=12)
            ax3.set_ylabel('Stagnation Count', fontsize=12)
            ax3.set_title('Stagnation & Restart Analysis', fontsize=13, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 子图4: 变异强度动态调整
            ax4 = axes[1, 0]
            mut_len = min(len(mutation_intensity_history), len(generations))
            gens_mut = generations[:mut_len]
            mut_vals = mutation_intensity_history[:mut_len]

            ax4.plot(gens_mut, mut_vals, 'purple',
                     linewidth=2.5, marker='v', markersize=3)
            if len(gens_mut) > 0:
                where_enhanced = np.array(mut_vals) >= 1.0
                if np.any(where_enhanced):
                    ax4.fill_between(gens_mut, 1.0, mut_vals,
                                     where=where_enhanced, alpha=0.3, color='purple',
                                     label='Enhanced Zone')
            ax4.axhline(y=1.0, color='gray', linestyle='-',
                        alpha=0.5, label='Base Intensity')
            ax4.set_xlabel('Generation', fontsize=12)
            ax4.set_ylabel('Mutation Intensity', fontsize=12)
            ax4.set_title('Adaptive Mutation Intensity',
                          fontsize=13, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # 子图5: 评估效率分析
            ax5 = axes[1, 1]
            eval_len = min(len(evaluation_count_history), len(generations))
            gens_eval = generations[:eval_len]
            eval_counts = evaluation_count_history[:eval_len]

            if len(eval_counts) > 0:
                eval_per_gen = [eval_counts[i] - (eval_counts[i-1] if i > 0 else 0)
                                for i in range(len(eval_counts))]

                if len(gens_eval) > 0:
                    ax5.bar(gens_eval, eval_per_gen, alpha=0.7,
                            color='skyblue', label='Evaluations per Gen')
                    if len(eval_per_gen) > 0:
                        avg_eval = np.mean(eval_per_gen)
                        ax5.axhline(y=avg_eval, color='red', linestyle='--',
                                    label=f'Average: {avg_eval:.0f}/gen')
            ax5.set_xlabel('Generation', fontsize=12)
            ax5.set_ylabel('Number of Evaluations', fontsize=12)
            ax5.set_title('Computational Efficiency Analysis',
                          fontsize=13, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 子图6: 优化总结统计
            ax6 = axes[1, 2]
            ax6.axis('off')

            # 计算统计信息
            total_runtime = time.time() - optimization_start
            total_improvements = len(improvement_generations)
            final_fitness = best_fitness_history[-1]
            initial_fitness = best_fitness_history[0]
            improvement_rate = ((final_fitness - initial_fitness) /
                                initial_fitness * 100) if initial_fitness > 0 else 0
            total_evaluations = evaluation_count_history[-1]
            avg_cache_hit = np.mean(cache_hit_ratios) * \
                100 if cache_hit_ratios else 0

            # 获取CPU信息
            try:
                # 尝试从/proc/cpuinfo获取真实CPU型号
                with open('/proc/cpuinfo', 'r') as f:
                    cpu_info = "Unknown CPU"
                    for line in f:
                        if 'model name' in line:
                            cpu_info = line.split(':')[1].strip()
                            break
            except:
                try:
                    cpu_info = platform.processor() or platform.machine()
                    if not cpu_info or cpu_info in ['x86_64', 'AMD64', 'i386']:
                        cpu_info = "Unknown CPU"
                except:
                    cpu_info = "Unknown CPU"

            # 获取内存信息
            try:
                with open('/proc/meminfo', 'r') as f:
                    mem_total = 0
                    for line in f:
                        if 'MemTotal' in line:
                            mem_kb = int(line.split()[1])
                            mem_gb = round(mem_kb / 1024 / 1024)
                            mem_info = f"{mem_gb}GB RAM"
                            break
                    else:
                        mem_info = "Unknown RAM"
            except:
                mem_info = "Unknown RAM"

            stats_text = f"""Optimization Summary
            
┌─────────────────────────────────┐
│  Final Coverage: {final_fitness:.3f} sec      │
│  Initial Coverage: {initial_fitness:.3f} sec    │
│  Improvement: {improvement_rate:+.1f}%             │
│  Improvements: {total_improvements} times            │
│  Total Runtime: {total_runtime:.2f} sec         │
│  Total Evaluations: {total_evaluations}           │
│  Avg Cache Hit Rate: {avg_cache_hit:.1f}%     │
│  Population Restarts: {len(restart_generations)}        │
│  Final Stagnation: {stagnation_history[-1]}         │
└─────────────────────────────────┘

System & Algorithm Config:
• CPU: {cpu_info[:25]}
• Memory: {mem_info}
• Processes: {self.n_processes}
• Population: {self.population_size}
• Max Gen: {self.generations}
• Stagnation: {self.stagnation_threshold}
            """

            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=11,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

            plt.tight_layout()
            plt.savefig('tmp/enhanced_ga_analysis.png', dpi=800, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.show()

            # 重置样式
            plt.style.use('default')

        if self.best_individual:
            result = {
                'drones': self.best_individual,
                'duration': self.best_fitness,
                'targeted_missile_ids': self.targeted_missile_ids
            }

            self.save_result_to_file(result)
            print(
                f"Performance: {self.evaluations} evaluations, {self.cache_hits} cache hits")
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

        cover_intervals = self.global_system.get_cover_intervals_all_jammers(
            self.targeted_missile_ids)
        missile_seconds = self.global_system.get_cover_seconds_all_jammers(
            self.targeted_missile_ids)

        with open(f'output/log/fastest_optimization_results_{self.Qname}.txt', 'a', encoding='utf-8') as f:
            f.write(f"最快GA优化结果 - {timestamp}\n")
            f.write(f"总覆盖时长: {result['duration']:.3f}秒\n")
            f.write(f"性能统计: {self.evaluations} 次评估, {self.cache_hits} 次缓存命中\n")
            f.write(
                f"缓存命中率: {self.cache_hits/(self.evaluations+self.cache_hits)*100:.1f}%\n")

            f.write(f"各导弹覆盖情况:\n")
            for missile_id in self.targeted_missile_ids:
                duration = missile_seconds.get(missile_id, 0.0)
                f.write(f"  {missile_id}: {duration:.3f}秒\n")
                intervals = cover_intervals.get(missile_id, [])
                if intervals:
                    for i, (start, end) in enumerate(intervals):
                        f.write(
                            f"    区间{i+1}: {start:.2f}s - {end:.2f}s (持续: {end-start:.2f}s)\n")
                else:
                    f.write("    无有效遮挡时间间隔\n")

            f.write(f"无人机参数:\n")
            for drone_id, drone_data in result['drones'].items():
                f.write(f"  {drone_id}:\n")
                f.write(
                    f"    速度: [{drone_data[0]:.2f}, {drone_data[1]:.2f}, 0.00]\n")
                f.write(f"    干扰弹参数:\n")
                for i, (father_t, smoke_delay) in enumerate(drone_data[2]):
                    f.write(
                        f"      干扰弹{i+1}: 发射时间={father_t:.2f}s, 烟雾延迟={smoke_delay:.2f}s\n")
            f.write("-" * 50 + "\n")

        print(
            f"最快GA结果已保存到 output/fastest_optimization_results_{self.Qname}.txt")
