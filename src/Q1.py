import numpy as np
import json
from Problem_object import Global_System_Q123


def find_cover_seconds_Q1():
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/drones_forward_vector-Q1.json") as f:
        drones_forward_vector = json.load(f)

    global_sys = Global_System_Q123(initial_positions, drones_forward_vector)
    global_sys.add_jammers(1, 1.5, 3.6)

    covered_times = []
    test_times = np.arange(7.7, 9.5, 0.01)
    for t in test_times:
        result = global_sys.detect_occlusion_single_missile_jammer(
            t, global_sys.Missiles['M1'], global_sys.jammers['FY1'][0])
        print(f"t={t:.2f}s: occlusion detected = {result}")
        if result:
            covered_times.append(t)
    print(f"\nCovered time periods: {covered_times}")


def test_Q1(tmp_time=7.9):
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/drones_forward_vector-Q1.json") as f:
        drones_forward_vector = json.load(f)

    global_sys = Global_System_Q123(initial_positions, drones_forward_vector)
    global_sys.add_jammers(1, 1.5, 3.6)

    result = global_sys.detect_occlusion_single_missile_jammer(
        tmp_time, global_sys.Missiles['M1'], global_sys.jammers['FY1'][0])
    print(f"t={tmp_time:.2f}s: occlusion detected = {result}")
    global_sys.virtualize_single_missile_jammer(
        tmp_time, global_sys.Missiles['M1'], global_sys.Drones['FY1'], global_sys.jammers['FY1'][0])


if __name__ == '__main__':
    # find_cover_seconds_Q1()
    test_Q1()
