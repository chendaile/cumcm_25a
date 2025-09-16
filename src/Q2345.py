"""Convenience wrappers for running the official scenarios."""

from __future__ import annotations

from Lets_Optimize import Lets_optimize


def Q2():
    return Lets_optimize(drone_ids=["FY1"], n_jammers=1, population_size=200,
                         generations=80, Qname="Q2", targeted_missile_ids=["M1"],
                         random_seed=123)


def Q3():
    return Lets_optimize(drone_ids=["FY1"], n_jammers=3, population_size=150,
                         generations=200, Qname="Q3", targeted_missile_ids=["M1"])


def Q4():
    return Lets_optimize(drone_ids=["FY1", "FY2", "FY3"], n_jammers=1,
                         population_size=500, generations=120, Qname="Q4",
                         targeted_missile_ids=["M1"], random_seed=1234)


def Q5():
    return Lets_optimize(drone_ids=["FY1", "FY2", "FY3", "FY4", "FY5"],
                         n_jammers=3, population_size=300, generations=120,
                         Qname="Q5", targeted_missile_ids=["M1", "M2", "M3"],
                         random_seed=2025)


def Q5_help1():
    return Lets_optimize(drone_ids=["FY1"], n_jammers=3, population_size=120,
                         generations=150, Qname="Q5_FY1",
                         targeted_missile_ids=["M1", "M2", "M3"])


def Q5_help2():
    return Lets_optimize(drone_ids=["FY2"], n_jammers=3, population_size=150,
                         generations=150, Qname="Q5_FY2",
                         targeted_missile_ids=["M1", "M2", "M3"])


def Q5_help3():
    return Lets_optimize(drone_ids=["FY3"], n_jammers=3, population_size=150,
                         generations=150, Qname="Q5_FY3",
                         targeted_missile_ids=["M1", "M2", "M3"])


def Q5_help4():
    return Lets_optimize(drone_ids=["FY4"], n_jammers=1, population_size=120,
                         generations=150, Qname="Q5_FY4", targeted_missile_ids=["M1"])


def Q5_help5():
    return Lets_optimize(drone_ids=["FY5"], n_jammers=1, population_size=120,
                         generations=150, Qname="Q5_FY5",
                         targeted_missile_ids=["M1", "M2", "M3"])


if __name__ == "__main__":
    Q5()
