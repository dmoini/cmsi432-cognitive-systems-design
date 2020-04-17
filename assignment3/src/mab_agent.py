'''
  mab_agent.py
  
  Agent specifications implementing Action Selection Rules.
'''

import numpy as np

# ----------------------------------------------------------------
# MAB Agent Superclasses
# ----------------------------------------------------------------


class MAB_Agent:
    '''
    MAB Agent superclass designed to abstract common components
    between individual bandit players (below)
    '''

    def __init__(self, K):
        # Number of actions/arms
        self.K = K
        # list of tuples of (action taken, reward received)
        # {r_t: [a_t_losses, a_t_wins]}
        self.history = {k: [1, 1] for k in range(self.K)}

    def give_feedback(self, a_t, r_t):
        '''
        Provides the action a_t and reward r_t chosen and received
        in the most recent trial, allowing the agent to update its
        history
        '''
        self.history[a_t][int(r_t == 1)] += 1

    def clear_history(self):
        '''
        IMPORTANT: Resets your agent's history between simulations.
        No information is allowed to transfer between each of the N
        repetitions
        '''
        # assume each action begins with 1 loss and 1 win so that P_R of each action is 0.5
        self.history = {k: [1, 1] for k in range(self.K)}

    def greedy_choice(self):
        P_R = [self.history[k][1] /
               sum(self.history[k]) for k in range(self.K)]
        max_prob_indexes = [i for i in range(self.K) if P_R[i] == max(P_R)]
        return np.random.choice(max_prob_indexes)

    def random_choice(self):
        return np.random.choice(list(range(self.K)))

        # ----------------------------------------------------------------
        # MAB Agent Subclasses
        # ----------------------------------------------------------------


class Greedy_Agent(MAB_Agent):
    '''
    Greedy bandit player that, at every trial, selects the
    arm with the presently-highest sampled Q value
    '''

    def __init__(self, K):
        MAB_Agent.__init__(self, K)

    def choose(self, *args):
        return self.greedy_choice()


class Epsilon_Greedy_Agent(MAB_Agent):
    '''
    Exploratory bandit player that makes the greedy choice with
    probability 1-epsilon, and chooses randomly with probability
    epsilon
    '''

    def __init__(self, K, epsilon):
        MAB_Agent.__init__(self, K)
        self.epsilon = epsilon

    def choose(self, *args):
        random_epsilon = np.random.rand()
        # Random Approach
        if random_epsilon < self.epsilon:
            return self.random_choice()
        # Greedy approach
        else:
            return self.greedy_choice()


class Epsilon_First_Agent(MAB_Agent):
    '''
    Exploratory bandit player that takes the first epsilon*T
    trials to randomly explore, and thereafter chooses greedily
    '''

    def __init__(self, K, epsilon, T):
        MAB_Agent.__init__(self, K)
        self.n = T * epsilon

    def choose(self, *args):
        # Explore: Random choice for the first n trials
        if self.n > 0:
            self.n -= 1
            return self.random_choice()
        # Exploit: Greedy choice ever after
        else:
            return self.greedy_choice()


class Epsilon_Decreasing_Agent(MAB_Agent):
    '''
    Exploratory bandit player that acts like epsilon-greedy but
    with a decreasing value of epsilon over time
    '''

    def __init__(self, K, epsilon, cooling_schedule=1, delta=0.0001, T=1000):
        MAB_Agent.__init__(self, K)
        self.epsilon = epsilon
        self.cooling_schedule = cooling_schedule
        # the amount by which to change epsilon if using a linear or exponential cooling schedule
        self.delta = delta
        self.trials = 0
        self.T = T

    # First cooling schedule: linear decrease in epsilon by difference of delta after each choice.
    def update_epsilon_linear(self):
        self.epsilon -= self.delta
        return self.epsilon

    # Second cooling schedule: exponential decay of epsilon by a factor of delta after each choice.
    def update_epsilon_exponential(self):
        self.epsilon *= (1 - self.delta)
        return self.epsilon

    # Third cooling schedule: logarithmic decay of epsilon according to the equation T(e) = -e * log(2 * currentTrials/totalTrials)
    def update_epsilon_logarithmic(self):
        self.trials += 1
        return -self.epsilon * np.log10(2 * self.trials/self.T)

    def get_current_temperature(self):
        if self.cooling_schedule == 1:
            return self.update_epsilon_linear()
        if self.cooling_schedule == 2:
            return self.update_epsilon_exponential()
        if self.cooling_schedule == 3:
            return self.update_epsilon_logarithmic()

    def choose(self, *args):
        # Calculates the current temperature based on the selected cooling schedule
        # Simultaneously updates self.epsilon or self.trials as needed
        proba_temperature = self.get_current_temperature()
        random_temperature = np.random.rand()
        # Random Approach
        if random_temperature < proba_temperature:
            return self.random_choice()
        # Greedy approach
        else:
            return self.greedy_choice()


class TS_Agent(MAB_Agent):
    '''
    Thompson Sampling bandit player that self-adjusts exploration
    vs. exploitation by sampling arm qualities from successes
    summarized by a corresponding beta distribution
    '''

    '''
    The steps of Thompson Sampling are thus (roughly):
        1. Sample an estimated reward "quality" from each arms' distributions.
        2. Choose the highest of those samples as the next arm chosen.
        3. Update history of outcome, shrinking variance of expected reward around arm chosen.
        4. Repeat.
    '''

    def __init__(self, K):
        MAB_Agent.__init__(self, K)

    def choose(self, *args):
        beta_results = [0, 0, 0, 0]
        for arm in self.history:
            l, w = self.history[arm]
            beta_results[arm] = np.random.beta(w, l)
        best_choice = beta_results.index(max(beta_results))
        return best_choice
