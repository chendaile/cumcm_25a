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

        print(f"最快GA版本 - {self.n_processes}进程并行 + 缓存 + 向量化")

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
            return avg_fitness

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
        
        return avg_fitness

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
        self.initialize_population_vectorized()

        # 收集详细统计信息
        best_fitness_history = []
        avg_fitness_history = []
        cache_hit_ratios = []
        stagnation_history = []
        mutation_intensity_history = []
        evaluation_count_history = []
        improvement_generations = []

        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")

            # 记录进化前状态
            old_best = self.best_fitness
            
            avg_fitness = self.evolve_generation_vectorized(generation)

            # 记录统计信息
            best_fitness_history.append(self.best_fitness)
            if avg_fitness is not None:
                avg_fitness_history.append(avg_fitness)
            else:
                avg_fitness_history.append(self.best_fitness)  # 重启时使用最佳值
            stagnation_history.append(self.stagnation_counter)
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
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('遗传算法优化详细分析', fontsize=16, fontweight='bold')

            # 子图1: 适应度收敛曲线
            ax1 = axes[0, 0]
            generations = range(1, len(best_fitness_history)+1)
            ax1.plot(generations, best_fitness_history, 'r-', linewidth=2.5, 
                    label='最佳适应度', marker='o', markersize=2)
            ax1.plot(generations, avg_fitness_history, 'b--', linewidth=1.5, 
                    label='平均适应度', alpha=0.7)
            ax1.fill_between(generations, avg_fitness_history, best_fitness_history, 
                           alpha=0.2, color='green', label='适应度差距')
            
            # 标记改进点
            if improvement_generations:
                improvement_values = [best_fitness_history[i-1] for i in improvement_generations]
                ax1.scatter(improvement_generations, improvement_values, 
                          color='gold', s=50, zorder=5, label='改进点', marker='*')
            
            ax1.set_xlabel('代数', fontsize=12)
            ax1.set_ylabel('覆盖时长 (秒)', fontsize=12)
            ax1.set_title('适应度收敛分析', fontsize=13, fontweight='bold')
            ax1.legend(loc='lower right')
            ax1.grid(True, alpha=0.3)

            # 子图2: 缓存效率分析
            ax2 = axes[0, 1]
            if cache_hit_ratios:
                ax2.plot(generations, [r*100 for r in cache_hit_ratios], 
                        'g-', linewidth=2, marker='s', markersize=3)
                ax2.fill_between(generations, 0, [r*100 for r in cache_hit_ratios], 
                               alpha=0.3, color='green')
                final_ratio = cache_hit_ratios[-1] * 100
                ax2.axhline(y=final_ratio, color='red', linestyle='--', 
                          label=f'最终命中率: {final_ratio:.1f}%')
                ax2.set_ylabel('缓存命中率 (%)', fontsize=12)
            ax2.set_xlabel('代数', fontsize=12)
            ax2.set_title('缓存效率变化', fontsize=13, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 子图3: 停滞与重启分析
            ax3 = axes[0, 2]
            ax3.plot(generations, stagnation_history, 'orange', linewidth=2, 
                    marker='d', markersize=3, label='停滞计数')
            ax3.axhline(y=self.stagnation_threshold, color='red', linestyle='--', 
                      label=f'重启阈值: {self.stagnation_threshold}')
            
            # 标记重启点
            restart_points = [i+1 for i, s in enumerate(stagnation_history) if s == 0 and i > 0]
            if restart_points:
                ax3.scatter(restart_points, [0]*len(restart_points), 
                          color='red', s=80, marker='X', zorder=5, label='种群重启')
            
            ax3.set_xlabel('代数', fontsize=12)
            ax3.set_ylabel('停滞计数', fontsize=12)
            ax3.set_title('停滞与重启分析', fontsize=13, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 子图4: 变异强度动态调整
            ax4 = axes[1, 0]
            ax4.plot(generations, mutation_intensity_history, 'purple', 
                    linewidth=2.5, marker='v', markersize=3)
            ax4.fill_between(generations, 1.0, mutation_intensity_history, 
                           where=[m >= 1.0 for m in mutation_intensity_history],
                           alpha=0.3, color='purple', label='强度增强区域')
            ax4.axhline(y=1.0, color='gray', linestyle='-', alpha=0.5, label='基准强度')
            ax4.set_xlabel('代数', fontsize=12)
            ax4.set_ylabel('变异强度', fontsize=12)
            ax4.set_title('自适应变异强度', fontsize=13, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # 子图5: 评估效率分析
            ax5 = axes[1, 1]
            eval_per_gen = [evaluation_count_history[i] - (evaluation_count_history[i-1] if i > 0 else 0) 
                           for i in range(len(evaluation_count_history))]
            ax5.bar(generations, eval_per_gen, alpha=0.7, color='skyblue', label='每代评估次数')
            avg_eval = np.mean(eval_per_gen)
            ax5.axhline(y=avg_eval, color='red', linestyle='--', 
                      label=f'平均: {avg_eval:.0f}次/代')
            ax5.set_xlabel('代数', fontsize=12)
            ax5.set_ylabel('评估次数', fontsize=12)
            ax5.set_title('计算效率分析', fontsize=13, fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

            # 子图6: 优化总结统计
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # 计算统计信息
            total_improvements = len(improvement_generations)
            final_fitness = best_fitness_history[-1]
            initial_fitness = best_fitness_history[0]
            improvement_rate = ((final_fitness - initial_fitness) / initial_fitness * 100) if initial_fitness > 0 else 0
            total_evaluations = evaluation_count_history[-1]
            avg_cache_hit = np.mean(cache_hit_ratios) * 100 if cache_hit_ratios else 0
            
            stats_text = f"""优化统计摘要
            
┌─────────────────────────────────┐
│  最终覆盖时长: {final_fitness:.3f} 秒        │
│  初始覆盖时长: {initial_fitness:.3f} 秒        │
│  改进幅度: {improvement_rate:+.1f}%           │
│  改进次数: {total_improvements} 次              │
│  总评估次数: {total_evaluations} 次           │
│  平均缓存命中率: {avg_cache_hit:.1f}%      │
│  种群重启次数: {len(restart_points)} 次        │
│  最终停滞计数: {stagnation_history[-1]} 次     │
└─────────────────────────────────┘

算法性能特征:
• 进程数: {self.n_processes}
• 种群大小: {self.population_size}
• 最大代数: {self.generations}
• 停滞阈值: {self.stagnation_threshold}
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
