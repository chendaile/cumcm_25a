import json
from Problem_object import Global_System_Q123
from Virtualizer import virtualize_single_jammer


def optimize_Q2():
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/drones_forward_vector-Q1.json") as f:
        drones_forward_vector = json.load(f)

    global_sys = Global_System_Q123(initial_positions, drones_forward_vector)

    print("Starting optimization for Q2...")
    print("Optimizing single missile with single drone and multiple jammers")

    best_params = global_sys.optimize_single_missile_drone_all_jammers(
        drone_id='FY1', n_jammers=1, population_size=30, generations=100, plot_convergence=True)

    if best_params:
        print(f"\nOptimization completed!")
        print(f"Best coverage duration: {best_params['duration']:.2f} seconds")
        print(
            f"Optimal velocity: [{best_params['velocity'][0]:.1f}, {best_params['velocity'][1]:.1f}, 0]")
        print(f"Jammer parameters:")
        for i, (father_t, smoke_delay) in enumerate(best_params['jammers']):
            print(
                f"  Jammer {i+1}: release_t={father_t:.2f}s, smoke_delay={smoke_delay:.2f}s")

        # Apply best parameters and test
        global_sys.reset_jammers('FY1')
        global_sys.update_drone_velocity('FY1', best_params['velocity'])

        for father_t, smoke_delay in best_params['jammers']:
            global_sys.add_jammers(1, father_t, smoke_delay)

        # Verify result
        final_duration = global_sys.get_cover_seconds_all_jammers()
        print(f"\nVerification: {final_duration:.2f} seconds coverage")

        # Visualize one moment
        virtualize_single_jammer(
            8.0, global_sys.Missiles['M1'], global_sys.Drones['FY1'],
            global_sys.jammers['FY1'][0] if global_sys.jammers['FY1'] else None,
            global_sys.true_goal)

    else:
        print("Optimization failed to find valid parameters")


if __name__ == '__main__':
    optimize_Q2()
