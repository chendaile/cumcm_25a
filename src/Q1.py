import json
from Problem_object import Global_System
from Virtualizer import virtualize_single_jammer


def find_cover_seconds_Q1():
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/initial_drones_forward_vector-Q1.json") as f:
        drones_forward_vector = json.load(f)

    global_sys = Global_System(initial_positions, drones_forward_vector)
    global_sys.add_jammers('FY1', 1.5, 3.6)

    duration = global_sys.get_cover_seconds_all_jammers()
    print(f"Total coverage duration: {duration:.2f} seconds")


def test_Q1(tmp_time=7.9):
    with open("data-bin/initial_positions.json") as f:
        initial_positions = json.load(f)
    with open("data-bin/drones_forward_vector-Q1.json") as f:
        drones_forward_vector = json.load(f)

    global_sys = Global_System(initial_positions, drones_forward_vector)
    global_sys.add_jammers('FY1', 1.5, 3.6)

    result = global_sys.detect_occlusion_single_jammer(
        tmp_time, global_sys.Missiles['M1'], global_sys.jammers['FY1'][0])
    print(f"t={tmp_time:.2f}s: occlusion detected = {result}")
    virtualize_single_jammer(
        tmp_time, global_sys.Missiles['M1'], global_sys.Drones['FY1'], global_sys.jammers['FY1'][0],
        global_sys.true_goal)


if __name__ == '__main__':
    # find_cover_seconds_Q1()
    test_Q1()
