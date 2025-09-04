import json
from Problem_object import Global_System_Q123
from Virtualizer import virtualize_all_jammers


def optimize_Q23():
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/drones_forward_vector.json") as f:
        drones_forward_vector = json.load(f)
    global_sys = Global_System_Q123(initial_positions, drones_forward_vector)
    print("Starting optimization ")

    best_params = global_sys.optimize_single_missile_drone_all_jammers(
        drone_id='FY1', n_jammers=1, population_size=250, generations=50, plot_convergence=True)

    if best_params:
        print(f"\nOptimization completed!")
        print(f"Best coverage duration: {best_params['duration']:.2f} seconds")
        print(
            f"Optimal velocity: [{best_params['velocity'][0]:.1f}, {best_params['velocity'][1]:.1f}, 0]")
        print(f"Jammer parameters:")
        for i, (father_t, smoke_delay) in enumerate(best_params['jammers']):
            print(
                f"  Jammer {i+1}: release_t={father_t:.2f}s, smoke_delay={smoke_delay:.2f}s")

        test(global_sys, best_params)
    else:
        print("Optimization failed to find valid parameters")


def test(global_sys, best_params):
    global_sys.reset_jammers('FY1')
    global_sys.update_drone_velocity('FY1', best_params['velocity'])
    for father_t, smoke_delay in best_params['jammers']:
        global_sys.add_jammers('FY1', father_t, smoke_delay)
    final_duration = global_sys.get_cover_seconds_all_jammers()
    cover_intervals = global_sys.get_cover_intervals_all_jammers()
    print(f"\nVerification: {final_duration:.2f} seconds coverage")
    print(f"Coverage intervals:")
    for i, (start, end) in enumerate(cover_intervals):
        print(
            f"  Interval {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")
    virtualize_all_jammers(
        8.0, global_sys.Missiles['M1'], global_sys.Drones['FY1'],
        global_sys.jammers['FY1'], global_sys.true_goal)


if __name__ == '__main__':
    # with open("data-bin/initial_positions.json") as f:
    #     initial_positions = json.load(f)
    # with open("data-bin/drones_forward_vector.json") as f:
    #     drones_forward_vector = json.load(f)
    # global_sys = Global_System_Q123(initial_positions, drones_forward_vector)
    # best_params = {
    #     'velocity': [-89.249, 0, 0],
    #     'jammers': [(1.239, 0)]
    # }
    # test(global_sys, best_params)
    optimize_Q23()
