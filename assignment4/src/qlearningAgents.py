# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math,numpy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_values = util.Counter() # dictionary of tuple (state, action):q_value, with default value of 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.q_values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        return self.getMaxActionAndValue(state)[1]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        return self.getMaxActionAndValue(state)[0]

    def getMaxActionAndValue(self, state):
        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return (None, 0)
        best_actions = []
        best_value = float("-inf")
        for action in actions:
            current_value = self.getQValue(state, action)
            if current_value > best_value: 
                best_actions = [action]
                best_value = current_value
            elif current_value == best_value:
                best_actions.append(action)
        best_action = random.choice(best_actions) 
        return (best_action, best_value) 

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
            return None
        coin_flip = util.flipCoin(self.epsilon)
        if coin_flip:
            return random.choice(legalActions)
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Use Q-Learning Update Rule
        sample = reward + self.discount * self.getMaxActionAndValue(nextState)[1]
        old_q_estimate = self.getQValue(state, action)
        sample_nudge = self.alpha * (sample - old_q_estimate)
        result = old_q_estimate + sample_nudge
        self.q_values[(state, action)] = result

    def getPolicy(self, state):
        print "getPolicy"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        print "getValue"
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector = [0.1 0.2 0.5] \dot [f1_value f2_value f3_value]
          where * is the dotProduct operator
        """
        weights = self.getWeights() # dictionary of {'feature name 1': weight1, 'feature name 2': weight2}
        featuresAndValues = self.featExtractor.getFeatures(state, action) # dictionary of {'feature name 1': value1, 'feature name 2': value2}
        weightVector, featureVector = [], []
        for featureName in featuresAndValues:
            featureVector.append(featuresAndValues[featureName])
            weightVector.append(weights[featureName])
        estimatedQValue = numpy.dot(weightVector, featureVector)
        return estimatedQValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        difference = reward + (self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        featuresAndValues = self.featExtractor.getFeatures(state, action) 
        for featureName, currentWeight in self.weights.items():
            self.weights[featureName] = currentWeight + self.alpha * difference * featuresAndValues[featureName]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            pass
