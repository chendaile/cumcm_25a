import numpy as np
from numba import njit

"""Default units are m and m/s"""


@njit
def check_occlusion_numba(missile_pos, target_pos, smoke_pos, smoke_radius=10):
    missile_to_target = target_pos - missile_pos
    missile_to_smoke = smoke_pos - missile_pos

    if np.dot(missile_to_smoke, missile_to_target) <= 0:
        return False

    target_norm = np.linalg.norm(missile_to_target)
    proj_length = np.dot(missile_to_smoke, missile_to_target) / target_norm
    proj_point = missile_pos + proj_length * missile_to_target / target_norm

    distance = np.linalg.norm(smoke_pos - proj_point)
    return distance <= smoke_radius


class Missile():
    def __init__(self, init_pos: np.ndarray):
        self.init_pos = init_pos
        self.missile_speed = 300
        self.forward_vector = -self.init_pos / np.linalg.norm(self.init_pos) * \
            self.missile_speed

    def get_pos(self, global_t):
        pos = self.init_pos + global_t * self.forward_vector
        if pos[0] < 0:
            raise ValueError("smoke_release_delay")
        return pos


class True_goal():
    def __init__(self, solid_bottom_enter_pos: np.ndarray):
        self.bottom_center_pos = solid_bottom_enter_pos
        self.radius = 7
        self.height = 10


class Jammer():
    def __init__(self, father_t, release_point: np.ndarray,
                 forward_vector: np.ndarray, smoke_release_delay):
        self.father_t = father_t
        self.forward_vector = forward_vector
        self.release_point = release_point
        self.Gravity = np.array([0, 0, -9.8])

        self.smoke_release_delay = smoke_release_delay
        self.smoke = Smoke(self.father_t + self.smoke_release_delay,
                           self.get_pos(operate_t=self.smoke_release_delay))

    def get_pos(self, global_t=None, operate_t=None):
        if operate_t is None:
            operate_t = global_t - self.father_t
        if operate_t < 0:
            raise ValueError("operate_t must be positive")
        pos = self.release_point + 1/2 * self.Gravity * operate_t**2 + \
            self.forward_vector * operate_t
        if pos[2] < 0:
            raise ValueError("Jammers shouldn't be underground.")
        return pos


class Smoke:
    def __init__(self, father_t, release_point: np.ndarray):
        self.father_t = father_t
        self.radius = 10
        self.forward_vector = np.array([0, 0, -3])
        self.smoke_duration = 20
        self.release_point = release_point

    def get_pos(self, global_t=None, operate_t=None):
        if operate_t is None:
            operate_t = global_t - self.father_t
        if operate_t < 0:
            raise ValueError("operate_t must be positive")
        pos = self.release_point + operate_t * self.forward_vector
        if pos[2] < 0:
            raise ValueError("Smoke shouldn't be underground.")
        return pos


class Drone:
    def __init__(self, init_pos: np.ndarray, forward_vector: np.ndarray):
        self.init_pos = init_pos
        self.forward_vector = forward_vector
        self.velocity_scalar = np.linalg.norm(self.forward_vector)
        if self.velocity_scalar > 140 or self.velocity_scalar < 70:
            raise ValueError(
                "The drone's speed must be between 70 and 140 m/s.")

    def get_pos(self, global_t):
        return self.init_pos + global_t * self.forward_vector

    def create_jammer(self, jammer_release_delay, smoke_release_delay):
        return Jammer(jammer_release_delay, self.get_pos(jammer_release_delay),
                      self.forward_vector, smoke_release_delay)


class Global_System_Q123:
    """Single messile and single drone"""

    def __init__(self, initial_positions: dict, drones_forward_vector: dict):
        self.Drones = {f'FY{str(i)}': Drone(
            np.array(initial_positions['drones'][f'FY{str(i)}']),
            np.array(drones_forward_vector[f'FY{str(i)}'])) for i in [1]}
        self.jammers = {f'FY{str(i)}': [] for i in [1]}
        self.Missiles = {f'M{str(i)}': Missile(
            np.array(initial_positions['missiles'][f'M{str(i)}'])) for i in [1]}
        self.true_goal = True_goal(
            np.array(initial_positions['target']['true_target']))

    def add_jammers(self, index, jammer_release_delay, smoke_release_delay):
        self.jammers[f'FY{str(index)}'].append(
            self.Drones[f'FY{str(index)}'].create_jammer(
                jammer_release_delay, smoke_release_delay))

    def check_occlusion(self, missile_pos, target_pos, smoke_pos, smoke_radius=10):
        return check_occlusion_numba(missile_pos, target_pos, smoke_pos, smoke_radius)

    def detect_occlusion_single_jammer(self, global_t, missile, jammer):
        missile_pos = missile.get_pos(global_t)
        if global_t < jammer.smoke.father_t:
            return False
        smoke_operate_t = global_t - jammer.smoke.father_t
        if smoke_operate_t > jammer.smoke.smoke_duration:
            return False

        smoke_pos = jammer.smoke.get_pos(global_t)
        target_bottom = self.true_goal.bottom_center_pos
        target_top = target_bottom + \
            np.array([0, 0, self.true_goal.height])
        occlusion_points = [
            target_bottom + np.array([self.true_goal.radius, 0, 0]),
            target_bottom + np.array([0, self.true_goal.radius, 0]),
            target_bottom + np.array([-self.true_goal.radius, 0, 0]),
            target_bottom + np.array([0, -self.true_goal.radius, 0]),
            target_bottom +
            np.array([self.true_goal.radius, 0, self.true_goal.height/2]),
            target_bottom +
            np.array([0, self.true_goal.radius, self.true_goal.height/2]),
            target_bottom +
            np.array([-self.true_goal.radius, 0, self.true_goal.height/2]),
            target_bottom +
            np.array([0, -self.true_goal.radius, self.true_goal.height/2]),
            target_top + np.array([self.true_goal.radius, 0, 0]),
            target_top + np.array([0, self.true_goal.radius, 0]),
            target_top + np.array([-self.true_goal.radius, 0, 0]),
            target_top + np.array([0, -self.true_goal.radius, 0])
        ]

        for point in occlusion_points:
            if not self.check_occlusion(missile_pos, point, smoke_pos):
                return False
        return True

    def detect_occlusion_all_jammers(self, global_t, missile, jammers_list):
        for jammer in jammers_list:
            if self.detect_occlusion_single_jammer(global_t, missile, jammer):
                return True
        return False

    def get_cover_seconds_all_jammers(self):
        covered_times = []
        test_times = np.arange(5.1, 20, 0.05)
        for t in test_times:
            result = self.detect_occlusion_all_jammers(
                t, self.Missiles['M1'], self.jammers['FY1'])
            if result:
                covered_times.append(t)

        if not covered_times:
            return 0.0
        return covered_times[-1] - covered_times[0]

    def update_drone_velocity(self, drone_id, velocity_vector):
        self.Drones[drone_id].forward_vector = np.array(velocity_vector)
        velocity_scalar = np.linalg.norm(velocity_vector)
        self.Drones[drone_id].velocity_scalar = velocity_scalar
        if velocity_scalar < 70 or velocity_scalar > 140:
            return False
        return True

    def reset_jammers(self, drone_id):
        self.jammers[drone_id] = []

    def optimize_single_missile_drone_all_jammers(self, drone_id='FY1', n_jammers=1, population_size=50, generations=100, plot_convergence=False):
        import random
        import matplotlib.pyplot as plt

        def create_individual():
            velocity_x = random.uniform(-140, 140)
            velocity_y = random.uniform(-140, 140)
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
            for _ in range(n_jammers):
                father_t = random.uniform(0.0, 10.0)
                smoke_delay = random.uniform(0.0, 10.0)
                jammers.append((father_t, smoke_delay))

            return [velocity_x, velocity_y, jammers]

        def evaluate_individual(individual):
            self.reset_jammers(drone_id)
            self.update_drone_velocity(
                drone_id, [individual[0], individual[1], 0])

            for father_t, smoke_delay in individual[2]:
                self.add_jammers(1, father_t, smoke_delay)

            return self.get_cover_seconds_all_jammers()

        def crossover(parent1, parent2):
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

        def mutate(individual):
            mutation_rate = 0.25 if generation < generations // 2 else 0.15

            if random.random() < mutation_rate:
                noise_scale = max(5, 30 - generation * 25 / generations)
                individual[0] += random.gauss(0, noise_scale)
                individual[1] += random.gauss(0, noise_scale)

                velocity_magnitude = np.sqrt(
                    individual[0]**2 + individual[1]**2)
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
                    noise_t = max(0.1, 0.5 - generation * 0.4 / generations)
                    noise_delay = max(
                        0.1, 0.6 - generation * 0.5 / generations)

                    new_father_t = max(0.5, min(3.0,
                                                individual[2][i][0] + random.gauss(0, noise_t)))
                    new_smoke_delay = max(2.0, min(5.0,
                                                   individual[2][i][1] + random.gauss(0, noise_delay)))

                    individual[2][i] = (new_father_t, new_smoke_delay)

            return individual

        population = [create_individual() for _ in range(population_size)]
        best_fitness_history = []
        avg_fitness_history = []
        best_individual = None
        best_fitness = 0

        for generation in range(generations):
            fitnesses = []
            for individual in population:
                fitness = evaluate_individual(individual)
                fitnesses.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
                    print(
                        f"Generation {generation+1}: New best {fitness:.3f}s")

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitnesses))

            population_with_fitness = list(zip(population, fitnesses))
            population_with_fitness.sort(key=lambda x: x[1], reverse=True)

            new_population = []
            elite_size = population_size // 4
            for i in range(elite_size):
                new_population.append(population_with_fitness[i][0])

            while len(new_population) < population_size:
                if sum(fitnesses) > 0:
                    tournament_size = 3
                    tournament1 = random.choices(
                        list(zip(population, fitnesses)), k=tournament_size)
                    parent1 = max(tournament1, key=lambda x: x[1])[0]

                    tournament2 = random.choices(
                        list(zip(population, fitnesses)), k=tournament_size)
                    parent2 = max(tournament2, key=lambda x: x[1])[0]
                else:
                    parent1 = random.choice(population)
                    parent2 = random.choice(population)
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)

            population = new_population

        if plot_convergence:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, generations+1), best_fitness_history,
                     'r-', linewidth=2, label='Best Fitness')
            plt.plot(range(1, generations+1), avg_fitness_history,
                     'b--', alpha=0.7, label='Average Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Coverage Duration (s)')
            plt.title('Genetic Algorithm Optimization Convergence')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('output/genetic_algorithm_convergence.png',
                        dpi=150, bbox_inches='tight')
            plt.show()

        if best_individual:
            return {
                'velocity': [best_individual[0], best_individual[1], 0],
                'jammers': best_individual[2],
                'duration': best_fitness
            }

        return None
