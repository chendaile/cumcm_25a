import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""Default units are m and m/s"""


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
        self.Gravity = np.array([0, 0, -10])

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


class Global_System_Q1:
    def __init__(self, initial_positions: dict, drones_forward_vector: dict):
        self.Drones = {f'FY{str(i)}': Drone(
            np.array(initial_positions['drones'][f'FY{str(i)}']),
            np.array(drones_forward_vector[f'FY{str(i)}'])) for i in [1]}
        self.Missiles = {f'M{str(i)}': Missile(
            np.array(initial_positions['missiles'][f'M{str(i)}'])) for i in [1]}
        self.true_goal = True_goal(
            np.array(initial_positions['target']['true_target']))


def check_occlusion(missile_pos, target_pos, smoke_pos, smoke_radius):
    missile_to_target = target_pos - missile_pos
    missile_to_smoke = smoke_pos - missile_pos

    if np.dot(missile_to_smoke, missile_to_target) <= 0:
        return False

    proj_length = np.dot(missile_to_smoke, missile_to_target) / \
        np.linalg.norm(missile_to_target)
    proj_point = missile_pos + proj_length * \
        missile_to_target / np.linalg.norm(missile_to_target)

    distance = np.linalg.norm(smoke_pos - proj_point)
    return distance <= smoke_radius


def detect_occlusion_Q1(global_t, missile, jammer, true_goal):
    missile_pos = missile.get_pos(global_t)

    if global_t < jammer.smoke.father_t:
        return False

    smoke_operate_t = global_t - jammer.smoke.father_t
    if smoke_operate_t > jammer.smoke.smoke_duration:
        return False

    smoke_pos = jammer.smoke.get_pos(global_t)

    target_bottom = true_goal.bottom_center_pos
    target_top = target_bottom + \
        np.array([0, 0, true_goal.height])
    target_center = (target_bottom + target_top) / 2

    occlusion_points = [
        target_bottom,
        target_center,
        target_top
    ]

    occluded_count = 0
    for point in occlusion_points:
        if check_occlusion(missile_pos, point, smoke_pos, jammer.smoke.radius):
            occluded_count += 1

    return occluded_count >= 2


def virtualize_Q1(global_t, M1, FY1, jammer1, true_goal):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    missile_pos = M1.get_pos(global_t)
    drone_pos = FY1.get_pos(global_t)

    ax.scatter(*missile_pos, color='red', s=100, label='M1 Missile')
    ax.scatter(*drone_pos, color='blue', s=100, label='FY1 Drone')

    target_pos = true_goal.bottom_center_pos
    ax.scatter(*target_pos, color='green', s=200,
               marker='s', label='True Target')

    if global_t >= jammer1.smoke.father_t and (global_t - jammer1.smoke.father_t) <= jammer1.smoke.smoke_duration:
        smoke_pos = jammer1.smoke.get_pos(global_t)

        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_smoke = jammer1.smoke.radius * \
            np.outer(np.cos(u), np.sin(v)) + smoke_pos[0]
        y_smoke = jammer1.smoke.radius * \
            np.outer(np.sin(u), np.sin(v)) + smoke_pos[1]
        z_smoke = jammer1.smoke.radius * \
            np.outer(np.ones(np.size(u)), np.cos(v)) + smoke_pos[2]

        ax.plot_surface(x_smoke, y_smoke, z_smoke,
                        alpha=0.4, color='orange')
        ax.scatter(*smoke_pos, color='red', s=200,
                   marker='o', label='Smoke Center')

        missile_to_smoke = smoke_pos - missile_pos
        smoke_distance = np.linalg.norm(missile_to_smoke)

        if smoke_distance > jammer1.smoke.radius:
            missile_to_smoke_unit = missile_to_smoke / smoke_distance

            perp1 = np.array([1, 0, 0]) if abs(
                missile_to_smoke_unit[0]) < 0.9 else np.array([0, 1, 0])
            perp1 = perp1 - \
                np.dot(perp1, missile_to_smoke_unit) * \
                missile_to_smoke_unit
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(missile_to_smoke_unit, perp1)

            sin_alpha = jammer1.smoke.radius / smoke_distance
            cos_alpha = np.sqrt(1 - sin_alpha**2)

            cone_points = []
            cone_theta = np.linspace(0, 2*np.pi, 36)
            for theta_val in cone_theta:
                tangent_dir = cos_alpha * missile_to_smoke_unit + sin_alpha * \
                    (np.cos(theta_val) * perp1 + np.sin(theta_val) * perp2)
                cone_end = missile_pos + 15000 * tangent_dir
                cone_points.append(cone_end)
                ax.plot([missile_pos[0], cone_end[0]], [missile_pos[1], cone_end[1]], [
                        missile_pos[2], cone_end[2]], 'r-', alpha=0.2)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title(f'Smoke Jamming Visualization at t={global_t}s')

    plt.tight_layout()
    plt.savefig(
        f'output/visualization_t{global_t}.png', dpi=800, bbox_inches='tight')
    plt.show()


def main_Q1():
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/drones_forward_vector-Q1.json") as f:
        drones_forward_vector = json.load(f)
    global_sys = Global_System_Q1(initial_positions, drones_forward_vector)
    # 此时我们之分析M1以及FY1
    M1 = global_sys.Missiles['M1']
    FY1 = global_sys.Drones['FY1']
    jammer1 = FY1.create_jammer(1.5, 3.6)
    true_goal = True_goal(
        np.array(initial_positions['target']['true_target']))
    print(f"Jammer released at t={jammer1.father_t}s")
    print(f"Smoke activated at t={jammer1.smoke.father_t}s")

    result = detect_occlusion_Q1(5, M1, jammer1, true_goal)
    print(f"occlusion detected: {result}")
    virtualize_Q1(8, M1, FY1, jammer1, true_goal)

    result = detect_occlusion_Q1(13, M1, jammer1, true_goal)
    print(f"occlusion detected: {result}")
    virtualize_Q1(8, M1, FY1, jammer1, true_goal)


if __name__ == '__main__':
    main_Q1()
