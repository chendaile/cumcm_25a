import json
from Problem_object import Global_System
from Virtualizer import virtualize_all_jammers, photography


def optimize_single_missile(drone_ids, n_jammers, population_size, generations, Qname):
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/initial_drones_forward_vector.json") as f:
        drones_forward_vector = json.load(f)
    global_sys = Global_System(initial_positions, drones_forward_vector)
    print("Starting optimization ")

    best_params = global_sys.optimize_single_missile_drone_all_jammers(
        drone_ids, n_jammers, population_size, generations, plot_convergence=True, Qname=Qname)

    if best_params:
        test(global_sys, best_params)
    else:
        print("Optimization failed to find valid parameters")


def test(best_params, video=False):
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/initial_drones_forward_vector.json") as f:
        drones_forward_vector = json.load(f)
    global_sys = Global_System(initial_positions, drones_forward_vector)

    for drone_id, drone_data in best_params['drones'].items():
        global_sys.reset_jammers(drone_id)
        global_sys.update_drone_velocity(
            drone_id, [drone_data[0], drone_data[1], 0])
        for father_t, smoke_delay in drone_data[2]:
            global_sys.add_jammers(drone_id, father_t, smoke_delay)

    final_duration = global_sys.get_cover_seconds_all_jammers(
        best_params['targeted_missile_ids'])
    cover_intervals = global_sys.get_cover_intervals_all_jammers(
        best_params['targeted_missile_ids'])
    # print(f"\nVerification: {final_duration:.2f} seconds coverage")
    # print(f"Coverage intervals:")
    # for i, (start, end) in enumerate(cover_intervals):
    #     print(
    #         f"  Interval {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")

    if video:
        all_jammers = []
        for drone_id in best_params['drones']:
            all_jammers.extend(global_sys.jammers[drone_id])
        active_drones = {
            drone_id: global_sys.Drones[drone_id] for drone_id in best_params['drones']}
        photography(global_sys.Missiles, active_drones,
                    all_jammers, global_sys.true_goal)


if __name__ == '__main__':
    best_params = {
        'drones': {
            "FY3": [0, 70, [(40, 2)]]
        },
        'targeted_missile_ids': ['M1']
    }
    test(best_params)
