import numpy as np
from numba import njit
from GA import GeneticOptimizer

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


class Global_System:
    """Single messile and single drone"""

    def __init__(self, initial_positions: dict, drones_forward_vector: dict):
        self.Drones = {}
        self.jammers = {}
        for drone_id in initial_positions['drones']:
            if drone_id in drones_forward_vector:
                self.Drones[drone_id] = Drone(
                    np.array(initial_positions['drones'][drone_id]),
                    np.array(drones_forward_vector[drone_id]))
                self.jammers[drone_id] = []

        self.Missiles = {f'M{str(i)}': Missile(
            np.array(initial_positions['missiles'][f'M{str(i)}'])) for i in [1]}
        self.true_goal = True_goal(
            np.array(initial_positions['target']['true_target']))

    def add_jammers(self, drone_id, jammer_release_delay, smoke_release_delay):
        self.jammers[drone_id].append(
            self.Drones[drone_id].create_jammer(
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
        intervals = self.get_cover_intervals_all_jammers()
        if not intervals:
            return 0.0
        return sum(end - start for start, end in intervals)

    def get_cover_intervals_all_jammers(self):
        covered_times = []
        test_times = np.arange(5.1, 20, 0.01)
        for t in test_times:
            all_jammers = []
            for drone_id in self.jammers:
                all_jammers.extend(self.jammers[drone_id])
            result = self.detect_occlusion_all_jammers(
                t, self.Missiles['M1'], all_jammers)
            if result:
                covered_times.append(t)

        if not covered_times:
            return []
        intervals = []
        start = covered_times[0]
        prev = covered_times[0]
        for t in covered_times[1:]:
            if t - prev > 0.1:
                intervals.append((start, prev))
                start = t
            prev = t
        intervals.append((start, prev))
        return intervals

    def update_drone_velocity(self, drone_id, velocity_vector):
        self.Drones[drone_id].forward_vector = np.array(velocity_vector)
        velocity_scalar = np.linalg.norm(velocity_vector)
        self.Drones[drone_id].velocity_scalar = velocity_scalar
        if velocity_scalar < 70 or velocity_scalar > 140:
            return False
        return True

    def reset_jammers(self, drone_id):
        self.jammers[drone_id] = []

    def optimize_single_missile_drone_all_jammers(self, drone_ids, n_jammers,
                                                  population_size,
                                                  generations, plot_convergence, Qname):
        optimizer = GeneticOptimizer(
            self, drone_ids, n_jammers, population_size, generations, Qname)
        return optimizer.optimize(plot_convergence)
