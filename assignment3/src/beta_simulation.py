import numpy as np


def beta_simulation(w, l):
    print(f'Wins: {w}, Losses: {l}')
    arr = [np.random.beta(w, l) for _ in range(10)]
    arr_avg = np.average(arr)
    arr_var = np.var(arr)

    print(f'Array: {arr}')
    print(f'Array average: {arr_avg}')
    print(f'Array variance: {arr_var}')
    print()


beta_simulation(4, 6)
beta_simulation(40, 60)
beta_simulation(400, 600)
