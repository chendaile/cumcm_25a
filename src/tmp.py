import json
import numpy as np
import random
from Problem_object import Global_System
from itertools import product
import matplotlib.pyplot as plt


class InitialSolutionGenerator:
    def __init__(self, global_system, drone_ids, n_jammers):
        self.global_system = global_system
        self.drone_ids = drone_ids
        self.n_jammers = n_jammers
        self.best_solutions = []

    def analyze_missile_trajectory(self):
        """分析导弹轨迹，找到关键时间点"""
        missile = self.global_system.Missiles['M1']
        target = self.global_system.true_goal.bottom_center_pos

        # 计算导弹到达目标的时间
        missile_to_target_distance = np.linalg.norm(target - missile.init_pos)
        arrival_time = missile_to_target_distance / missile.missile_speed

        # 关键时间段：导弹接近目标前的一段时间
        critical_start = max(5.0, arrival_time - 15.0)  # 最早5秒开始
        critical_end = min(20.0, arrival_time - 2.0)    # 到达前2秒结束

        print(f"导弹到达时间: {arrival_time:.2f}s")
        print(f"关键干扰时间段: {critical_start:.2f}s - {critical_end:.2f}s")

        return critical_start, critical_end, arrival_time

    def calculate_optimal_intercept_positions(self, critical_time):
        """计算在关键时间点的最佳拦截位置"""
        missile = self.global_system.Missiles['M1']
        target = self.global_system.true_goal.bottom_center_pos

        # 导弹在关键时间的位置
        missile_pos = missile.get_pos(critical_time)

        # 导弹到目标的方向
        missile_to_target = target - missile_pos
        missile_to_target_unit = missile_to_target / \
            np.linalg.norm(missile_to_target)

        # 在导弹与目标之间选择几个关键拦截点
        intercept_distances = [0.3, 0.5, 0.7]  # 距离导弹位置的比例
        intercept_positions = []

        for ratio in intercept_distances:
            intercept_pos = missile_pos + ratio * missile_to_target
            intercept_positions.append(intercept_pos)

        return intercept_positions

    def calculate_required_velocity(self, drone_id, target_pos, target_time):
        """计算无人机到达指定位置所需的速度"""
        drone = self.global_system.Drones[drone_id]
        drone_init_pos = drone.init_pos

        # 计算需要的位移
        required_displacement = target_pos - drone_init_pos

        # 计算需要的速度
        required_velocity = required_displacement / target_time
        velocity_magnitude = np.linalg.norm(required_velocity)

        # 检查速度约束
        if velocity_magnitude < 70:
            required_velocity = required_velocity / velocity_magnitude * 70
            velocity_magnitude = 70
        elif velocity_magnitude > 140:
            required_velocity = required_velocity / velocity_magnitude * 140
            velocity_magnitude = 140

        return required_velocity, velocity_magnitude

    def generate_strategic_initial_solutions(self, num_solutions=20):
        """生成基于策略的初始解"""
        solutions = []
        critical_start, critical_end, arrival_time = self.analyze_missile_trajectory()

        # 策略1: 时间分散策略 - 不同无人机在不同时间释放干扰弹
        print("生成策略1: 时间分散策略")
        time_points = np.linspace(
            critical_start, critical_end, len(self.drone_ids))

        for i in range(num_solutions // 4):
            solution = {}
            for j, drone_id in enumerate(self.drone_ids):
                # 为每个无人机分配不同的时间点
                target_time = time_points[j] + random.uniform(-1.0, 1.0)
                target_time = max(1.0, min(15.0, target_time))

                # 计算拦截位置
                intercept_positions = self.calculate_optimal_intercept_positions(
                    target_time)
                target_pos = random.choice(intercept_positions)

                # 计算所需速度
                required_velocity, _ = self.calculate_required_velocity(
                    drone_id, target_pos, target_time)

                # 干扰弹参数
                smoke_delay = random.uniform(2.0, 4.0)

                solution[drone_id] = [
                    required_velocity[0], required_velocity[1],
                    [(target_time, smoke_delay)]
                ]

            solutions.append(solution)

        # 策略2: 空间分散策略 - 不同无人机覆盖不同空间区域
        print("生成策略2: 空间分散策略")
        for i in range(num_solutions // 4):
            solution = {}
            common_time = random.uniform(critical_start + 2, critical_end - 2)

            intercept_positions = self.calculate_optimal_intercept_positions(
                common_time)

            for j, drone_id in enumerate(self.drone_ids):
                # 为每个无人机分配不同的空间位置
                if j < len(intercept_positions):
                    target_pos = intercept_positions[j]
                else:
                    target_pos = random.choice(intercept_positions)

                # 添加随机偏移
                offset = np.random.normal(0, 200, 3)  # 200米标准差
                target_pos = target_pos + offset

                required_velocity, _ = self.calculate_required_velocity(
                    drone_id, target_pos, common_time)

                smoke_delay = random.uniform(2.0, 4.0)

                solution[drone_id] = [
                    required_velocity[0], required_velocity[1],
                    [(common_time, smoke_delay)]
                ]

            solutions.append(solution)

        # 策略3: 基于单机最优解的组合
        print("生成策略3: 基于单机最优解的组合")
        single_drone_solutions = self.generate_single_drone_solutions()

        for i in range(num_solutions // 4):
            solution = {}
            for drone_id in self.drone_ids:
                # 随机选择一个单机解作为基础
                base_solution = random.choice(single_drone_solutions)
                # 添加扰动
                velocity_x = base_solution[0] + random.uniform(-20, 20)
                velocity_y = base_solution[1] + random.uniform(-20, 20)
                father_t = base_solution[2] + random.uniform(-1, 1)
                smoke_delay = base_solution[3] + random.uniform(-0.5, 0.5)

                # 确保约束
                velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                if velocity_magnitude < 70:
                    scale = 70 / velocity_magnitude
                    velocity_x *= scale
                    velocity_y *= scale
                elif velocity_magnitude > 140:
                    scale = 140 / velocity_magnitude
                    velocity_x *= scale
                    velocity_y *= scale

                father_t = max(0.5, min(15.0, father_t))
                smoke_delay = max(1.0, min(5.0, smoke_delay))

                solution[drone_id] = [velocity_x,
                                      velocity_y, [(father_t, smoke_delay)]]

            solutions.append(solution)

        # 策略4: 随机但有约束的解
        print("生成策略4: 约束随机策略")
        for i in range(num_solutions - len(solutions)):
            solution = {}
            for drone_id in self.drone_ids:
                # 在合理范围内随机生成
                velocity_magnitude = random.uniform(70, 140)
                velocity_angle = random.uniform(0, 2 * np.pi)
                velocity_x = velocity_magnitude * np.cos(velocity_angle)
                velocity_y = velocity_magnitude * np.sin(velocity_angle)

                father_t = random.uniform(critical_start, critical_end)
                smoke_delay = random.uniform(2.0, 4.0)

                solution[drone_id] = [velocity_x,
                                      velocity_y, [(father_t, smoke_delay)]]

            solutions.append(solution)

        return solutions

    def generate_single_drone_solutions(self):
        """生成一些已知的单机较优解作为参考"""
        # 基于Q2和Q3的经验，但调整时间参数使其更早
        good_solutions = [
            [-140, 1.13, 1.5, 2.5],   # 早期释放版本
            [-120, 20, 2.0, 2.0],     # 偏向y方向，早期
            [-100, -30, 2.5, 1.8],    # 负y方向，早期
            [-130, 0, 1.8, 2.2],      # 纯x方向，早期
            [-110, 40, 2.2, 2.8],     # 高y速度，早期
            [-90, -20, 3.0, 1.5],     # 较晚但仍早期
            [-70, 0, 1.0, 2.5],       # 最低速度，很早
            [-140, -10, 2.8, 2.0]     # 高速低y，早期
        ]
        return good_solutions

    def evaluate_and_rank_solutions(self, solutions):
        """评估并排序解"""
        solution_fitness = []

        print(f"评估 {len(solutions)} 个初始解...")
        for i, solution in enumerate(solutions):
            try:
                # 重置系统
                for drone_id in self.drone_ids:
                    self.global_system.reset_jammers(drone_id)

                # 应用解
                for drone_id, drone_data in solution.items():
                    self.global_system.update_drone_velocity(
                        drone_id, [drone_data[0], drone_data[1], 0])
                    for father_t, smoke_delay in drone_data[2]:
                        self.global_system.add_jammers(
                            drone_id, father_t, smoke_delay)

                # 评估
                fitness = self.global_system.get_cover_seconds_all_jammers()
                solution_fitness.append((solution, fitness))

                if (i + 1) % 5 == 0:
                    print(
                        f"已评估 {i + 1}/{len(solutions)} 个解，当前最佳: {max(solution_fitness, key=lambda x: x[1])[1]:.3f}s")

            except Exception as e:
                print(f"解 {i} 评估失败: {e}")
                solution_fitness.append((solution, 0.0))

        # 按适应度排序
        solution_fitness.sort(key=lambda x: x[1], reverse=True)

        print(f"\n初始解评估完成!")
        print(f"最佳初始解覆盖时间: {solution_fitness[0][1]:.3f}s")
        print(f"前5名覆盖时间: {[f'{sf[1]:.3f}s' for sf in solution_fitness[:5]]}")

        return solution_fitness

    def save_best_initial_solutions(self, solution_fitness, filename="data-bin/q4_initial_solutions.json"):
        """保存最佳的初始解"""
        # 保存前10个最佳解
        best_solutions = []
        for solution, fitness in solution_fitness[:10]:
            # 转换为可JSON序列化的格式
            json_solution = {}
            for drone_id, drone_data in solution.items():
                json_solution[drone_id] = {
                    'velocity': [float(drone_data[0]), float(drone_data[1])],
                    'jammers': [[float(father_t), float(smoke_delay)]
                                for father_t, smoke_delay in drone_data[2]]
                }

            best_solutions.append({
                'solution': json_solution,
                'fitness': float(fitness)
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(best_solutions, f, indent=2, ensure_ascii=False)

        print(f"最佳初始解已保存到 {filename}")
        return best_solutions

    def visualize_solutions_distribution(self, solution_fitness):
        """可视化解的分布"""
        fitnesses = [sf[1] for sf in solution_fitness]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(fitnesses, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Coverage Duration (s)')
        plt.ylabel('Number of Solutions')
        plt.title('Initial Solutions Fitness Distribution')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(range(len(fitnesses)), fitnesses, 'b-', alpha=0.7)
        plt.xlabel('Solution Index (ranked)')
        plt.ylabel('Coverage Duration (s)')
        plt.title('Ranked Solutions Performance')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('tmp/q4_initial_solutions_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.show()


def generate_q4_initial_solutions():
    """主函数：为Q4生成初始解"""
    # 加载系统配置
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/initial_drones_forward_vector.json") as f:
        drones_forward_vector = json.load(f)

    # 创建系统
    global_sys = Global_System(initial_positions, drones_forward_vector)

    # 创建初始解生成器
    generator = InitialSolutionGenerator(
        global_sys,
        drone_ids=['FY1', 'FY2', 'FY3'],
        n_jammers=1
    )

    print("开始生成Q4的初始解...")

    # 生成策略性初始解
    solutions = generator.generate_strategic_initial_solutions(
        num_solutions=400)

    # 评估和排序
    solution_fitness = generator.evaluate_and_rank_solutions(solutions)

    # 保存最佳解
    best_solutions = generator.save_best_initial_solutions(solution_fitness)

    # 可视化
    generator.visualize_solutions_distribution(solution_fitness)

    return best_solutions


if __name__ == "__main__":
    best_solutions = generate_q4_initial_solutions()

    print("\n=== 最佳初始解详情 ===")
    for i, sol_data in enumerate(best_solutions[:3]):
        print(f"\n解 {i+1} (覆盖时间: {sol_data['fitness']:.3f}s):")
        for drone_id, drone_data in sol_data['solution'].items():
            velocity = drone_data['velocity']
            jammers = drone_data['jammers']
            print(f"  {drone_id}: 速度=[{velocity[0]:.1f}, {velocity[1]:.1f}], "
                  f"干扰弹=[t={jammers[0][0]:.2f}s, delay={jammers[0][1]:.2f}s]")
