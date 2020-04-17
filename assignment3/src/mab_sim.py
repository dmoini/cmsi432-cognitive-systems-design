'''
  mab_sim.py

  Simulation suite for comparing different Action Selection Rule
  performance on traditional Multi-Armed Bandit Problems
'''

import numpy as np
import plotly as py
import plotly.graph_objs as go
import multiprocessing
from plotly import subplots
from joblib import Parallel, delayed
from mab_agent import *

# ----------------------------------------------------------------
# Simulation Parameters
# ----------------------------------------------------------------

# Configure simulation
SIM_NAME = "cmsi_432_ass3_sims"
N = 1000    # Number of Monte Carlo Repetitions
T = 1000    # Trials per MC Repetition
N_CORES = multiprocessing.cpu_count()-1
np.random.seed(0)  # For reproducible results

# Reward Signal Probabilities
# P(R_t = 1 | do(A_t))
P_R = np.array(
    # A =  0     1     2     3
    [0.50, 0.60, 0.40, 0.30]
)
K = len(P_R)

# Initialize learning agents; change these based on what the spec
# asks you to plot on any given problem

# ONLY CHANGE THIS NUMBER, it will do the rest by itself. Don't comment and uncomment stuff!
# 0=greedy, 1=epsilon_greedy, 2=epsilon_first, 3=epsilon_decreasing, 4=ts, 5=all
sim_problem = 5

agents = [
    [
        Greedy_Agent(K),
        Greedy_Agent(K),
        Greedy_Agent(K)
    ],
    [
        Epsilon_Greedy_Agent(K, 0.05),
        Epsilon_Greedy_Agent(K, 0.1),
        Epsilon_Greedy_Agent(K, 0.15)
    ],
    [
        Epsilon_First_Agent(K, 0.1, T),
        Epsilon_First_Agent(K, 0.2, T),
        Epsilon_First_Agent(K, 0.3, T),
        Epsilon_First_Agent(K, 0.4, T),
        Epsilon_First_Agent(K, 0.5, T),
        Epsilon_First_Agent(K, 0.6, T),
        Epsilon_First_Agent(K, 0.7, T),
        Epsilon_First_Agent(K, 0.8, T),
        Epsilon_First_Agent(K, 0.9, T),
    ],
    [
        # cooling schedule 1, epsilon = 0.2, delta = 3 different tests cases
        Epsilon_Decreasing_Agent(K, 0.2, 1, 0.0005, T),
        Epsilon_Decreasing_Agent(K, 0.2, 1, 0.0010, T),
        Epsilon_Decreasing_Agent(K, 0.2, 1, 0.0015, T),
        # cooling schedule 2, epsilon = 0.2, delta = 3 different tests cases
        Epsilon_Decreasing_Agent(K, 0.2, 2, 0.0005, T),
        Epsilon_Decreasing_Agent(K, 0.2, 2, 0.0010, T),
        Epsilon_Decreasing_Agent(K, 0.2, 2, 0.0015, T),
        # cooling schedule 3, epsilon = 0.2, delta = 3 different tests cases
        Epsilon_Decreasing_Agent(K, 0.2, 3, 0.0005, T),
        Epsilon_Decreasing_Agent(K, 0.2, 3, 0.0010, T),
        Epsilon_Decreasing_Agent(K, 0.2, 3, 0.0015, T)
    ],
    [
        TS_Agent(K)
    ],
    [
        Greedy_Agent(K),
        Epsilon_Greedy_Agent(K, 0.1),
        Epsilon_First_Agent(K, 0.5, T),
        Epsilon_Decreasing_Agent(K, 0.2, 2, 0.0015, T),
        TS_Agent(K)
    ]

]
# Change these to describe the agents being compared in a given simulation
ag_names = [
    [
        "Greedy",
        "Greedy",
        "Greedy"
    ],
    [
        "Epsilon-Greedy (0.05)",
        "Epsilon-Greedy (0.1)",
        "Epsilon-Greedy (0.15)"
    ],
    [
        "Epsilon-First (0.1)",
        "Epsilon-First (0.2)",
        "Epsilon-First (0.3)",
        "Epsilon-First (0.4)",
        "Epsilon-First (0.5)",
        "Epsilon-First (0.6)",
        "Epsilon-First (0.7)",
        "Epsilon-First (0.8)",
        "Epsilon-First (0.9)",
    ],
    [
        "Epsilon-Decreasing-Linear (0.0005)",
        "Epsilon-Decreasing-Linear (0.001)",
        "Epsilon-Decreasing-Linear (0.0015)",
        "Epsilon-Decreasing-Exponential (0.0005)",
        "Epsilon-Decreasing-Exponential (0.001)",
        "Epsilon-Decreasing-Exponential (0.0015)",
        "Epsilon-Decreasing-Logarithmic (0.0005)",
        "Epsilon-Decreasing-Logarithmic (0.001)",
        "Epsilon-Decreasing-Logarithmic (0.0015)"
    ],
    [
        "TS"
    ],
    [
        "Greedy",
        "Epsilon-Greedy",
        "Epsilon-First (0.5)",
        "Epsilon-Decreasing-Exponential (0.0015)",
        "TS"
    ]
]
# Colors for the graphs (you don't need to change these unless you're an art snob)
ag_colors = [
    ('rgb(255, 0, 0)'),  # First agent is red
    ('rgb(220, 110, 0)'),  # Second agent is orange
    ('rgb(255, 165, 0)'),  # Third agent is gold
    ('rgb(130, 220, 0)'),  # Fourth agent is lime
    ('rgb(0, 128, 0)'),  # Fifth agent is dark green
    ('rgb(0, 175, 175)'),  # Sixth agent is turquoise
    ('rgb(0, 0, 255)'),  # Seventh agent is blue
    ('rgb(110, 0, 220)'),  # Eighth agent is purple
    ('rgb(255, 0, 255)'),  # Ninth agent is magenta
]
AG_COUNT = len(agents[sim_problem])


# ----------------------------------------------------------------
# Simulation Functions
# ----------------------------------------------------------------

def run_sim():
    '''
    Runs a single MC iteration of the simulation consisting of T trials.
    '''
    ag_reg = np.zeros((AG_COUNT, T))
    ag_opt = np.zeros((AG_COUNT, T))

    # Reset agent histories for this MC repetition
    for a in agents[sim_problem]:
        a.clear_history()

    for t in range(T):
        # Find the optimal action and reward rate for this t
        best_a_t = np.argmax(P_R)
        max_t = P_R[best_a_t]
        # Determine chosen action and reward for each agent
        # within this trial, t
        for a_ind, ag in enumerate(agents[sim_problem]):
            a_t = ag.choose()
            r_t = np.random.choice([0, 1], p=[1 - P_R[a_t], P_R[a_t]])
            ag.give_feedback(a_t, r_t)
            regret_t = max_t - r_t
            ag_reg[a_ind, t] += regret_t
            ag_opt[a_ind, t] += int(a_t == best_a_t)

    return [ag_reg, ag_opt]


def gen_graph(cum_reg, cum_opt, names, colors):
    '''
    Reporting mechanism that generates graphical reports on the
    probability that each agent takes the optimal action and the
    agent's cumulative regret, both as a function of the current
    trial
    '''
    AG_COUNT = cum_reg.shape[0]
    traces = []
    fig = subplots.make_subplots(rows=1, cols=2, subplot_titles=(
        'Probability of Optimal Action', 'Cumulative Regret'))
    fig['layout']['xaxis1'].update(title='Trial', range=[0, T])
    fig['layout']['xaxis2'].update(title='Trial', range=[0, T])
    fig['layout']['yaxis1'].update(title='Probability of Optimal Action')
    fig['layout']['yaxis2'].update(title='Cumulative Regret')

    # Plot cumulative regret
    for a in range(AG_COUNT):
        trace = go.Scatter(
            x=list(range(T)),
            y=cum_opt[a, :],
            line=dict(
                color=colors[a]
            ),
            name=names[a]
        )
        fig.append_trace(trace, 1, 1)

    # Plot optimal arm choice
    for a in range(AG_COUNT):
        trace = go.Scatter(
            x=list(range(T)),
            y=cum_reg[a, :],
            line=dict(
                color=colors[a]
            ),
            name="[REG]" + names[a],
            showlegend=False
        )
        fig.append_trace(trace, 1, 2)

    py.offline.plot(fig, filename=("./cum_reg_" + SIM_NAME + ".html"))


# ----------------------------------------------------------------
# Simulation Workhorse
# ----------------------------------------------------------------

if __name__ == "__main__":
    print("=== MAB Simulations Beginning ===")

    # Record-keeping data structures across MC simulations
    round_reg = np.zeros((AG_COUNT, T))
    round_opt = np.zeros((AG_COUNT, T))
    cum_reg = np.zeros((AG_COUNT, T))

    # MAIN WORKHORSE - Runs MC repetitions of simulations in parallel:
    sim_results = Parallel(n_jobs=N_CORES, verbose=1)(
        delayed(run_sim)() for i in range(N))
    for (ind, r) in enumerate(sim_results):
        round_reg += r[0]
        round_opt += r[1]

    # Reporting phase:
    for a in range(AG_COUNT):
        cum_reg[a] = np.array([np.sum(round_reg[a, 0:i+1]) for i in range(T)])
    cum_reg = cum_reg / N
    cum_opt = round_opt / N
    gen_graph(cum_reg, cum_opt, ag_names[sim_problem], ag_colors)

    print("[!] Simulations Completed")
