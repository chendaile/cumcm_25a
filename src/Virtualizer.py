import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')


def virtualize_all_jammers(global_t, missiles: dict, drones,
                           jammers, true_goal,
                           save_only=False, save_path=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for missile_id, missile in missiles.items():
        missile_pos = missile.get_pos(global_t)
        ax.scatter(*missile_pos, color='b', s=60,
                   label=missile_id + ' Missile')
        ax.text(missile_pos[0], missile_pos[1], missile_pos[2], missile_id,
                fontsize=10, color='b', fontweight='bold')

    target_pos = true_goal.bottom_center_pos
    ax.scatter(*target_pos, color='green', s=50, label='True Target')
    ax.text(target_pos[0], target_pos[1], target_pos[2], '  Target',
            fontsize=10, color='green', fontweight='bold')

    for drone_id, drone in drones.items():
        drone_pos = drone.get_pos(global_t)
        ax.scatter(*drone_pos, color='darkred', s=60,
                   label=f'{drone_id}')
        ax.text(drone_pos[0], drone_pos[1], drone_pos[2], f'  {drone_id}',
                fontsize=10, color='darkred', fontweight='bold')

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
            ax.text(smoke_pos[0]+50, smoke_pos[1], smoke_pos[2]-150, f'S{i+1}',
                    fontsize=10, color='orange', fontweight='bold')

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
                cone_theta = np.linspace(0, 2*np.pi, 150)
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

    ax.set_xlim([0, 22000])
    ax.set_ylim([-3500, 1500])
    ax.set_zlim([0, 2000])

    ax.legend()
    ax.set_title(
        f'Smoke Jamming Visualization at t={global_t}s ({active_smoke_count} active smokes)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=800, bbox_inches='tight')
    else:
        plt.savefig(
            f'tmp/visualization_t_{global_t}s.png', dpi=800, bbox_inches='tight')

    if save_only:
        plt.close(fig)
    else:
        plt.show()


def photography(missiles: dict, drones, jammers, true_goal,
                time_start=5, time_end=25.0, fps=5, output_dir='tmp/frames'):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    dt = 1.0 / fps
    time_points = np.arange(time_start, time_end + dt, dt)
    total_frames = len(time_points)
    print(f"开始摄影: {time_start}s - {time_end}s, {fps}fps, 共{total_frames}帧")
    for i, global_t in enumerate(time_points):
        frame_filename = f'{output_dir}/frame_{i:04d}_t_{global_t:.2f}s.png'
        virtualize_all_jammers(global_t, missiles, drones, jammers, true_goal,
                               save_only=True, save_path=frame_filename)
        if (i + 1) % (fps * 2) == 0:
            progress = (i + 1) / total_frames * 100
            print(f"进度: {progress:.1f}% ({i+1}/{total_frames} 帧)")

    print(f"摄影完成！共生成 {total_frames} 帧图片")
    print(f"图片保存在: {output_dir}")
    print(f"转换为视频的命令:")
    print(f"  # Create sequential links first in output directory:")
    print(
        f"  python3 -c \"import os,glob; files=sorted(glob.glob('{output_dir}/frame_*_t_*.png'), key=lambda x: float(x.split('_t_')[1].split('s.png')[0])); [os.symlink(f, f'seq_frame_{{i:04d}}.png') for i,f in enumerate(files)]\"")
    print(
        f"  ffmpeg -y -r {fps} -i seq_frame_%04d.png -vf scale=2942:2972 -c:v libx264 -pix_fmt yuv420p output/smoke_jamming_animation.mp4")
