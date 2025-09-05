from Optimize_single_missile import optimize_single_missile


def Q2():
    optimize_single_missile(drone_ids=['FY1'],
                            n_jammers=1,
                            population_size=60,
                            generations=200,
                            Qname='Q2')


def Q3():
    optimize_single_missile(drone_ids=['FY1'],
                            n_jammers=3,
                            population_size=150,
                            generations=400,
                            Qname='Q3')


def Q4():
    optimize_single_missile(drone_ids=['FY1', 'FY2', 'FY3'],
                            n_jammers=1,
                            population_size=150,
                            generations=200,
                            Qname='Q4')


if __name__ == "__main__":
    Q3()
