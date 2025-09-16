from cumcm.plan import DronePlan, JammerPlan, Plan
import numpy as np


def test_plan_clamp_orders_jammers():
    plan = Plan({
        "FY1": DronePlan(
            velocity=np.array([10.0, 0.0]),
            jammers=[
                JammerPlan(2.0, 1.0),
                JammerPlan(1.0, 2.0),
                JammerPlan(1.5, 1.5),
            ],
        )
    })
    plan.clamp()
    jams = plan.drones["FY1"].jammers
    assert all(jams[i].release_time < jams[i + 1].release_time for i in range(len(jams) - 1))
    assert all(j.release_time >= 0 for j in jams)
    assert plan.drones["FY1"].velocity.shape == (3,)
