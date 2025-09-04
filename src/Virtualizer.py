import numpy as np
import matplotlib.pyplot as plt


def virtualize_single_jammer(global_t, missile, drone, jammer, true_goal):
    """原单个干扰弹可视化函数，保持向后兼容性"""
    jammers = [jammer] if jammer else []
    virtualize_all_jammers(global_t, missile, drone, jammers, true_goal)


def virtualize_all_jammers(global_t, missile, drone, jammers, true_goal):
    """可视化所有干扰弹的函数"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    missile_pos = missile.get_pos(global_t)
    drone_pos = drone.get_pos(global_t)
    target_pos = true_goal.bottom_center_pos

    ax.scatter(*missile_pos, color='darkred', s=60, label='M1 Missile')
    ax.scatter(*drone_pos, color='blue', s=60, label='FY1 Drone')
    ax.scatter(*target_pos, color='green', s=50, label='True Target')

    colors = ['orange', 'yellow', 'cyan', 'magenta', 'lime']
    active_smoke_count = 0

    for i, jammer in enumerate(jammers):
        if global_t >= jammer.smoke.father_t and (global_t - jammer.smoke.father_t) <= jammer.smoke.smoke_duration:
            active_smoke_count += 1
            smoke_pos = jammer.smoke.get_pos(global_t)
            color = colors[i % len(colors)]

            u = np.linspace(0, 2 * np.pi, 50)
            v = np.linspace(0, np.pi, 50)
            x_smoke = jammer.smoke.radius * \
                np.outer(np.cos(u), np.sin(v)) + smoke_pos[0]
            y_smoke = jammer.smoke.radius * \
                np.outer(np.sin(u), np.sin(v)) + smoke_pos[1]
            z_smoke = jammer.smoke.radius * \
                np.outer(np.ones(np.size(u)), np.cos(v)) + smoke_pos[2]

            ax.plot_surface(x_smoke, y_smoke, z_smoke,
                            alpha=0.4, color=color)
            ax.scatter(*smoke_pos, color='orange', s=30,
                       marker='o', label=f'Smoke {i+1} Center')

            missile_to_smoke = smoke_pos - missile_pos
            smoke_distance = np.linalg.norm(missile_to_smoke)

            if smoke_distance > jammer.smoke.radius:
                missile_to_smoke_unit = missile_to_smoke / smoke_distance

                perp1 = np.array([1, 0, 0]) if abs(
                    missile_to_smoke_unit[0]) < 0.9 else np.array([0, 1, 0])
                perp1 = perp1 - \
                    np.dot(perp1, missile_to_smoke_unit) * \
                    missile_to_smoke_unit
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(missile_to_smoke_unit, perp1)

                sin_alpha = jammer.smoke.radius / smoke_distance
                cos_alpha = np.sqrt(1 - sin_alpha**2)

                cone_points = []
                cone_theta = np.linspace(0, 2*np.pi, 150)  # 减少线条数量避免过度拥挤
                for theta_val in cone_theta:
                    tangent_dir = cos_alpha * missile_to_smoke_unit + sin_alpha * \
                        (np.cos(theta_val) * perp1 + np.sin(theta_val) * perp2)
                    cone_end = missile_pos + 25000 * tangent_dir
                    cone_points.append(cone_end)
                    ax.plot([missile_pos[0], cone_end[0]], [missile_pos[1], cone_end[1]], [
                            missile_pos[2], cone_end[2]], color='red', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    ax.set_title(
        f'Smoke Jamming Visualization at t={global_t}s ({active_smoke_count} active smokes)')

    plt.tight_layout()
    plt.savefig(
        f'tmp/visualization_t_{global_t}s.png', dpi=800, bbox_inches='tight')
    plt.show()
