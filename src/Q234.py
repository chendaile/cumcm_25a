from Lets_Optimize import Lets_optimize


def Q2():
    Lets_optimize(drone_ids=['FY1'],
                  n_jammers=1,
                  population_size=60,
                  generations=200,
                  Qname='Q2')


def Q3():
    Lets_optimize(drone_ids=['FY1'],
                  n_jammers=3,
                  population_size=150,
                  generations=300,
                  Qname='Q3')


def Q4():
    Lets_optimize(drone_ids=['FY1', 'FY2', 'FY3'],
                  n_jammers=1,
                  population_size=150,
                  generations=5,
                  Qname='Q4')


def Q4_help1():
    Lets_optimize(drone_ids=['FY1'],
                  n_jammers=1,
                  population_size=150,
                  generations=10,
                  Qname='Q4_help1')


def Q4_help2():
    Lets_optimize(drone_ids=['FY2'],
                  n_jammers=1,
                  population_size=150,
                  generations=300,
                  Qname='Q4_help2')


def Q4_help3():
    Lets_optimize(drone_ids=['FY3'],
                  n_jammers=1,
                  population_size=150,
                  generations=300,
                  Qname='Q4_help3')


if __name__ == "__main__":
    # Q4_help1()
    Q4()
