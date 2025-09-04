import numpy as np
import json
import matplotlib.pyplot as plt
from Problem_object import Global_System_Q1, True_goal, check_occlusion


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
                cone_end = missile_pos + 25000 * tangent_dir
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
        f'output/visualization_t{global_t:.2f}.png', dpi=800, bbox_inches='tight')
    # plt.show()


def find_cover_seconds_Q1():
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/drones_forward_vector-Q1.json") as f:
        drones_forward_vector = json.load(f)
    global_sys = Global_System_Q1(initial_positions, drones_forward_vector)

    M1 = global_sys.Missiles['M1']
    FY1 = global_sys.Drones['FY1']
    jammer1 = FY1.create_jammer(1.5, 3.6)
    true_goal = True_goal(np.array(initial_positions['target']['true_target']))

    covered_times = []
    test_times = np.arange(7.7, 9.5, 0.01)

    for t in test_times:
        result = detect_occlusion_Q1(t, M1, jammer1, true_goal)
        print(f"t={t:.2f}s: occlusion detected = {result}")
        if result:
            covered_times.append(t)
            virtualize_Q1(t, M1, FY1, jammer1, true_goal)

    print(f"\nCovered time periods: {covered_times}")


def test_Q1():
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

    result = detect_occlusion_Q1(8.5, M1, jammer1, true_goal)
    print(f"occlusion detected: {result}")
    virtualize_Q1(8.5, M1, FY1, jammer1, true_goal)

    result = detect_occlusion_Q1(9, M1, jammer1, true_goal)
    print(f"occlusion detected: {result}")
    virtualize_Q1(9, M1, FY1, jammer1, true_goal)


if __name__ == '__main__':
    find_cover_seconds_Q1()
