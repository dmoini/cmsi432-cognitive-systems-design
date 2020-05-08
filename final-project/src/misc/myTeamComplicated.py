# myTeam.py
# ---------
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

import random
import time
import util
import math
import numpy
import json
from os import path
from copy import deepcopy

from captureAgents import CaptureAgent
from game import Directions
import game

from pprint import pprint

DEBUG_OFFENSE_ONLY = True
DEBUG_DEFENSE_ONLY = False

TRAINING = True
WEIGHTS_FILE = "weights_cece.json"
EMPTY_WEIGHTS = {
    "offense": {},
    "defense": {},
}
FEATURES = {
    'offense': [
        # shared
        'DistanceFromInitialState',
        'DistanceToNearestEnemy',
        'OnOffense',
        'ScaredGhost',
        'MovingTowardsNearestEnemy',
        'Score',
        'ChangedFromPacmanToGhost',
        'ChangedFromGhostToPacman',
        'IsPacman',
        'IsGhost',
        # offense-only
        'DistanceToNearestFood',
        'DistanceToNearestPowerCapsule',
        'StateHasFoodAtCurrentPosition',
        'StateHasPowerCapsuleAtCurrentPosition'
        'WillDepositFoodAtSuccessor'
    ],
    'defense': [
        # shared
        'DistanceFromInitialState',
        'DistanceToNearestEnemy',
        'OnOffense',
        'ScaredGhost',
        'MovingTowardsNearestEnemy',
        'Score',
        'ChangedFromPacmanToGhost',
        'ChangedFromGhostToPacman',
        'IsPacman',
        'IsGhost',
        # defense-only
        'DistanceToNearestDefendingFood',
        'NumberOfInvaders'
    ]
}

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveDummyAgent', second='DefensiveDummyAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class QLearningAgent(CaptureAgent):
    """
    Superclass containing mechanics for approximate Q-learning.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.start = gameState.getAgentPosition(self.index)

        # Either read from weights.json file if exists, or initialize weights to counter with all features set to 0
        # At the end of the game, if file did not exist, will write weights to file (only if TRAINING == True)
        isOffensiveAgent = isinstance(self, OffensiveDummyAgent)
        self.weights = self.initializeWeights(isOffensiveAgent) if TRAINING else self.readWeightsFromFile(isOffensiveAgent)
        # dictionary of 'featureName' : value, with defalt of 0

        # features = util.Counter()

        # LEARNING PARAMETERS
        self.discount = 0.1
        # learning rate; make large (> 0.1) for training and 0 for exploitation
        self.alpha = 0.1
        # probability that the agent will choose randomly instead of optimally; only relevant during training
        self.epsilon = 0.1
        self.livingReward = -0.1

        # MAINTAINING GAME INFORMATION
        self.lastAction = Directions.STOP
        # The value of this should be 76; any two positions in this maze can be at most 76 moves apart
        self.maxMazeDistance = max(self.distancer._distances.values())
        self.collectedFood = 0
        self.totalFood = len(self.getFood(gameState).asList())
        self.numberOfEnemies = len(self.getOpponents(gameState))
        self.justDied = 0

    def chooseAction(self, gameState):
        """
        Picks the best of all legal actions based on their estimated Q values, which are computed
        with a linear combination of the feature values and their weights.
        This is the function that is called at every turn; any other functions that should be called
        each turn should be called from here.
        """
        # Only update weights if we are currently training
        if TRAINING:
            # First, update weights based on reward received from the move we just took
            s_prime = self.getCurrentObservation()
            a = self.lastAction

            agentName = "Offensive Agent" if isinstance(self, OffensiveDummyAgent) else "Defensive Agent"
            showOutput = (DEBUG_OFFENSE_ONLY and agentName is "Offensive Agent") or (DEBUG_DEFENSE_ONLY and agentName is "Defensive Agent")
            if showOutput:
                print(agentName + " action just taken: " + str(a))

            s = self.getPreviousObservation() if len(
                self.observationHistory) > 1 else s_prime

            reward = self.getReward(s, a, s_prime) # sets self.justDied = 25
            self.updateWeights(s, a, s_prime, reward)

        # Choose our next action!
        actions = gameState.getLegalActions(self.index)

        qValuesOfNextActions = [self.evaluatePotentialNextState(
            gameState, a) for a in actions]
        maxValue = max(qValuesOfNextActions)
        bestActions = [a for a, v in zip(
            actions, qValuesOfNextActions) if v == maxValue]

        # If there are 2 (or fewer) pellets left, the game is pretty much over, so the best action will be 
        # the one that moves us closer to where we initially started
        foodLeft = len(self.getFood(gameState).asList())
        if foodLeft <= 2:
            actionChoice = self.getActionToGoBackHome(gameState, actions)
            self.lastAction = actionChoice

            agentName = "Offensive Agent" if isinstance(self, OffensiveDummyAgent) else "Defensive Agent"
            showOutput = (DEBUG_OFFENSE_ONLY and agentName is "Offensive Agent") or (DEBUG_DEFENSE_ONLY and agentName is "Defensive Agent")
            if showOutput:
                print(agentName + " CHOOSING ACTION: " + str(actionChoice))

            return actionChoice

        # Do a coin flip with probability self.epsilon to choose randomly instead of optimally, only if TRAINING
        coin_flip = util.flipCoin(self.epsilon)
        if coin_flip and TRAINING:
            actionChoice = random.choice(actions)
            self.lastAction = actionChoice

            agentName = "Offensive Agent" if isinstance(self, OffensiveDummyAgent) else "Defensive Agent"
            showOutput = (DEBUG_OFFENSE_ONLY and agentName is "Offensive Agent") or (DEBUG_DEFENSE_ONLY and agentName is "Defensive Agent")
            if showOutput:
                print(agentName + "CHOOSING ACTION: " + str(actionChoice))

            return actionChoice

        # In all other cases, choose the best action based on computed Q values
        # If multiple actions are tied, break ties randomly.
        actionChoice = random.choice(bestActions)
        self.lastAction = actionChoice

        agentName = "Offensive Agent" if isinstance(self, OffensiveDummyAgent) else "Defensive Agent"
        showOutput = (DEBUG_OFFENSE_ONLY and agentName is "Offensive Agent") or (DEBUG_DEFENSE_ONLY and agentName is "Defensive Agent")
        if showOutput:
            print(agentName + " CHOOSING ACTION: " + str(actionChoice))

        return actionChoice

    def getActionToGoBackHome(self, gameState, actions):
        bestDist = self.maxMazeDistance
        bestAction = None
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(self.index)
            dist = self.distancer.getDistance(self.start, pos2)
            if dist < bestDist:
                bestAction = action
                bestDist = dist
        return bestAction

    def getQValue(self, gameState, action):
        """
        Returns a number representing Q(gameState,action) = linear combination of all feature values and their weights
        = weightsVector * featureVector, where * is the dotProduct operator
        Example: [0.1 0.2 0.5] dotProduct [f1_value f2_value f3_value]
        """
        # weights is a Counter of {'feature name 1': weight1, 'feature name 2': weight2, ...}
        weights = self.getWeights()
        # featuresAndValues is a Counter of {'feature name 1': featureValue1, ...}
        featuresAndValues = self.getFeatures(gameState, action)

        # multiplication of two Counters (defined in util) should return the dot product of their elements
        
        estimatedQValue = weights * featuresAndValues

        # agentName = "Offensive Agent" if isinstance(self, OffensiveDummyAgent) else "Defensive Agent"
        # showOutput = (DEBUG_OFFENSE_ONLY and agentName is "Offensive Agent") or (DEBUG_DEFENSE_ONLY and agentName is "Defensive Agent")
        # if showOutput:
        #     print(agentName + " GET Q VALUE")
        #     print("-- WEIGHTS:  ")
        #     pprint(weights)
        #     print("-- FEATURES: ")
        #     pprint(featuresAndValues)
        #     print(" returning estimated Q value: " + str(estimatedQValue))

        return estimatedQValue
    
    def evaluatePotentialNextState(self, gameState, action):
        """
        Returns the Q-value of the next state s' that we would be at if we took
        the given action from our current gameState
        """
        return self.getQValue(gameState, action)

    def getWeights(self):
        """
        Returns a Counter containing feature names mapped to their weights.
        Note that this method does not take in the gameState or action because
        normally, weights do not depend on the gameState.
        """
        return self.weights

    def updateWeights(self, prevGameState, action, currentGameState, reward):
        """
        Updates self.weights based on the transition we JUST TOOK, i.e. the action that
        brought us from the previous state to the current state
        """
        # Difference =
        #    (Reward received just now + value of acting optimally from this state on)
        #    - (value of the state we were just in)
        difference = reward + (self.discount * self.computeValueFromQValues(
            currentGameState)) - self.getQValue(prevGameState, action)
        featuresAndValues = self.getFeatures(prevGameState, action)
        for featureName, currentWeight in self.weights.items():
            self.weights[featureName] = currentWeight + \
                self.alpha * difference * featuresAndValues[featureName]

        agentName = "Offensive Agent" if isinstance(self, OffensiveDummyAgent) else "Defensive Agent"
        showOutput = (DEBUG_OFFENSE_ONLY and agentName is "Offensive Agent") or (DEBUG_DEFENSE_ONLY and agentName is "Defensive Agent")
        if showOutput:
            print("-- update weights -- WEIGHTS:  ")
            pprint(self.weights)
            print("-- update weights -- FEATURES WE USED TO UPDATE WEIGHTS: ")
            # pprint(features)
            pprint(featuresAndValues)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        The gameState that this method returns is the same as the current one
        EXCEPT that the agent would have taken the specified action.
        We use this method for considering potential actions before choosing one.
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action), the maximum Q value over all
        legal actions that can be taken from the given state.
        Note that if there are no legal actions, this returns a value of 0.0.
        """
        return self.getMaxActionAndValue(state)[1]

    def getMaxActionAndValue(self, state):
        actions = state.getLegalActions(self.index)
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

    def final(self, gameState):
        """
        This is called at the end of the game. Overrides Capture Agent's final method.
        In order to avoid bugs, just copy pasted what was in Capture Agent's method
        instead of calling super().
        """
        print("### END OF GAME ###")
        # Copied from Capture Agent's final method
        self.observationHistory = []
        if TRAINING:
            self.writeWeightsToFile(self.weights, isinstance(self, OffensiveDummyAgent))

    ###########################
    # HANDING FILE READ/WRITE #
    ###########################

    def readWeightsFromFile(self, isOffensiveAgent):
        if not path.exists(WEIGHTS_FILE):
            raise Exception("{0} does not exist".format(WEIGHTS_FILE))
        f = open(WEIGHTS_FILE, 'r')
        weightsDict = json.load(f)
        f.close()
        return weightsDict

    def writeWeightsToFile(self, weights, isOffensiveAgent):
        updatedWeights = self.readWeightsFromFile(isOffensiveAgent) if path.exists(
            WEIGHTS_FILE) else deepcopy(EMPTY_WEIGHTS)
        updatedWeights['offense' if isOffensiveAgent else 'defense'] = weights
        f = open(WEIGHTS_FILE, 'w')
        json.dump(updatedWeights, f)
        f.close()

    def convertDictToCounter(self, d):
        result = util.Counter()
        for key, value in d.items():
            result[key] = value
        return result

    def generateDefaultWeights(self, isOffensiveAgent):
        weights = util.Counter()
        for feature in FEATURES['offense' if isOffensiveAgent else 'defense']:
            weights[feature] = 0.0
        return weights

    def initializeWeights(self, isOffensiveAgent):
        if path.exists(WEIGHTS_FILE):
            # self.readWeightsFromFile returns a dictionary containing two
            # weight dictionaries, one with key 'offense' and one with key 'defense
            weightsDict = self.readWeightsFromFile(isOffensiveAgent).get(
                'offense' if isOffensiveAgent else 'defense')
            weightsCounter = self.convertDictToCounter(weightsDict)
            return weightsCounter
        else:
            return self.generateDefaultWeights(isOffensiveAgent)


class DummyAgent(QLearningAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def getSharedFeatures(self, gameState, action):
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        # features['DistanceFromInitialState'] = self.normalizeDistanceFeature(
            # self.getDistanceFromInitialState(successor))
        
        normalizedDistanceToNearestEnemy = self.normalizeDistanceFeature(self.getDistanceToNearestEnemy(successor) + 1)
        features['DistanceToNearestEnemy'] = normalizedDistanceToNearestEnemy if normalizedDistanceToNearestEnemy < 1 else 0
        # features['OnOffense'] = self.isOnOffense(successor)
        
        
        # features['OnOffense'] = self.isPacman(successor)
        # features['ScaredGhost'] = self.isScared(successor)
        # features['MovingTowardsNearestEnemy'] = self.isMovingTowardsNearestEnemy(gameState, successor)
        # features['Score'] = self.getScore(successor)
        # features['ChangedFromPacmanToGhost'] = self.changedFromPacmanToGhost(
        #     gameState, successor)
        
        # features['ChangedFromGhostToPacman'] = self.changedFromGhostToPacman(
        #     gameState, successor)
        # features['IsPacman'] = float(self.isPacman(successor))
        # features['IsGhost'] = float(self.isGhost(successor))
        return features

    def getSharedReward(self, prevState, action, currentState):
        """
        Returns R(s, a, s') where s is the previous gameState, a is the action we just took,
        and s' is the current gameState.
        s = prevState
        a = action we just took (one of NORTH, SOUTH, ... etc.)
        s' = currentState
        """
        reward = self.livingReward

        if self.hasDied(prevState, currentState):
            print("AH FUCK WE JUST DIED")
            self.justDied = 1
            reward -= 100

        # if self.depositedFood(prevState, currentState):
        #     reward += self.collectedFood / 20.0
        #     self.collectedFood = 0.0

        # if self.gotCloserToEnemy(prevState, currentState):

        # If the Pacman is a scared ghost, run from the enemies
        # if self.isScared(currentState):
        #     distanceToNearestEnemy = self.getDistanceToNearestEnemy(
        #         currentState)
        #     reward -= 1.0 / distanceToNearestEnemy
        
        wentFurther = self.wentFurtherFromInitialState(prevState, currentState)
        distanceFromInitialState = self.getDistanceFromInitialState(
            currentState)
        if wentFurther:
            reward -= (2.0 / distanceFromInitialState) if distanceFromInitialState != 0 else 1

        return reward

    ####################################################################
    #                                                                  #
    #    *ALL* FEATURE EXTRACTOR HELPERS AND STATE-CHECKING HELPERS    #
    #                                                                  #
    #  These are feature extractors for ANY feature that either agent  #
    #  could use, placed in superclass for ease of access in case      # 
    #  either one needs to check something.                            #
    #  This way, the Offensive and Defensive agents will only differ   #
    #  in *WHICH* functions they choose to *CALL* but not which        #
    #  methods they have available.                                    #
    #                                                                  #
    ####################################################################

    #########################
    # Both Agents, Probably #
    #########################

    def getDistanceToNearestFood(self, state):
        foodList = self.getFood(state).asList()
        minDistance = self.maxMazeDistance
        if len(foodList) > 0:
            myPos = state.getAgentState(self.index).getPosition()
            minDistance = min([self.distancer.getDistance(myPos, food)]
                              for food in foodList)[0]
        return minDistance
        
    def getDistanceToNearestDefendingFood(self, state):
        foodList = self.getFoodYouAreDefending(state).asList()
        minDistance = self.maxMazeDistance
        if len(foodList) > 0:
            myPos = state.getAgentState(self.index).getPosition()
            minDistance = min([self.distancer.getDistance(myPos, food)]
                              for food in foodList)[0]
        return minDistance

    def getDistanceFromInitialState(self, state):
        initialPos = state.getInitialAgentPosition(self.index)
        currentPos = state.getAgentState(self.index).getPosition()
        distance = self.distancer.getDistance(initialPos, currentPos)
        return distance

    def getDistanceToNearestEnemy(self, state):
        currentPos = state.getAgentState(self.index).getPosition()
        # opponentIndices = self.getOpponents(state)
        # minDistance = self.maxMazeDistance
        # for opponent in opponentIndices:
        #     if state.getAgentPosition(opponent) == None:
        #         continue
        #     oppPos = state.getAgentPosition(opponent)
        #     distance = self.distancer.getDistance(currentPos, oppPos)
        #     if distance:
        #         minDistance = min(minDistance, distance)
        # return minDistance
        nearestEnemyIndex = self.getNearestEnemyIndex(state)
        opponentPos = state.getAgentPosition(nearestEnemyIndex) if (nearestEnemyIndex is not None) else None
        distance = self.distancer.getDistance(currentPos, opponentPos) if opponentPos else self.maxMazeDistance
        return distance

    def getNearestEnemyIndex(self, state):
        currentPos = state.getAgentState(self.index).getPosition()
        opponentIndices = self.getOpponents(state)
        # Starting the minimum distance at 76 because 76 is the maximum maze distance
        # between any two points. Want to avoid setting minDistance to inf because if
        # no opponents are visible with certainty, then it will just return inf for
        # this feature's value, which would be difficult for the weights to adjust to
        minDistance = self.maxMazeDistance
        nearestOpponentIndex = None
        for i in range(len(opponentIndices)):
            opponent = opponentIndices[i]
            if state.getAgentPosition(opponent) == None:
                continue
            opponentPos = state.getAgentPosition(opponent)
            distance = self.distancer.getDistance(currentPos, opponentPos)
            if distance and distance < minDistance:
                minDistance = distance
                nearestOpponentIndex = i
        return nearestOpponentIndex

    # def getIndexOfNearestEnemy(self, state):
    #     currentPos = state.getAgentState(self.index).getPosition()
    #     opponentIndices = self.getOpponents(state)
    #     minDistance = self.maxMazeDistance
    #     ghostIndex = -1
    #     for opponent in opponentIndices:
    #         if state.getAgentPosition(opponent) == None:
    #             continue
    #         oppPos = state.getAgentPosition(opponent)
    #         distance = self.distancer.getDistance(currentPos, oppPos)
    #         if distance < minDistance:
    #             minDistance = distance
    #             ghostIndex = opponent
    #     return ghostIndex

    def isMovingTowardsNearestEnemy(self, gameState, successor):
        movingTowards = self.gotCloserToEnemy(gameState, successor)
        return float(movingTowards)

    # TODO remove this, same as the feature "IsPacman"
    def isOnOffense(self, state):
        myPos = state.getAgentPosition(self.index)
        return float(state.getAgentState(self.index).isPacman)

    def stateHasFoodAtCurrentPosition(self, state):
        myPos = state.getAgentPosition(self.index)
        return float(state.hasFood(myPos[0], myPos[1]))

    def changedFromGhostToPacman(self, state1, state2):
        return float(self.isGhost(state1) and self.isPacman(state2))

    def changedFromPacmanToGhost(self, state1, state2):
        return float(self.isPacman(state1) and self.isGhost(state2))

    ##############################
    # Only for Offense, probably #
    ##############################

    def getDistanceToNearestPowerCapsule(self, state):
        capsuleList = self.getCapsules(state)
        minDistance = self.maxMazeDistance
        if len(capsuleList) > 0:
            myPos = state.getAgentState(self.index).getPosition()
            minDistance = min([self.distancer.getDistance(myPos, capsule)]
                              for capsule in capsuleList)[0]
        return minDistance

    def isPoweredUp(self, state):
        opponentIndices = self.getOpponents(state)
        for index in opponentIndices:
            if state.getAgentState(index).scaredTimer > 0:
                return True
        return False

    def atePowerCapsule(self, gameState, successor):
        return (not self.isPoweredUp(gameState)) and self.isPoweredUp(successor)

    def couldDieOnNextEnemyMove(self, state):
        print("COULD DIE ON NEXT ENEMY MOVE")
        if self.getDistanceToNearestEnemy(state) <= 1.01:
            print("ENEMY IS <= 1 MOVE AWAY")
            enemyIndex = self.getNearestEnemyIndex(state)
            print("EMENY IS A GHOST, NOT PACMAN: " + str(not state.getAgentState(enemyIndex).isPacman))
            return state.getAgentState(self.index).isPacman and not state.getAgentState(enemyIndex).isPacman
        return False

    def willDepositFoodAtSuccessor(self, gameState, successor):
        willDepositFood = self.collectedFood > 0 and self.changedFromPacmanToGhost(
            gameState, successor) == 1
        return float(willDepositFood)
        
    def hasPowerCapsuleAtPosition(self, gameState, pos1, pos2):
        powerCapsuleLocations = self.getCapsules(gameState)
        if len(powerCapsuleLocations) == 0:
            return False
        powerCapsulePos1, powerCapsulePos2 = powerCapsuleLocations[0]
        return powerCapsulePos1 == pos1 and powerCapsulePos2 == pos2

    ##############################
    # Only for Defense, probably #
    ##############################

    def hasEnemyDied(self, state1, state2):
        opponentIndices = self.getOpponents(state1)
        myPos = state1.getAgentState(self.index).getPosition()

        for index in opponentIndices:
            if state1.getAgentPosition(index) == None:
                continue
            enemyPosBefore = state1.getAgentState(index).getPosition()
            if self.distancer.getDistance(myPos, enemyPosBefore) <= 2 and state2.getAgentPosition(index) == None:
                return True
        return False

    def numberOfInvaders(self, gameState):
        enemies = [gameState.getAgentState(i)
                   for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        return len(invaders)

    #########################
    # MISCELLANEOUS HELPERS #
    #########################

    def gotCloserToEnemy(self, state1, state2):
        prevEnemyDistance = self.getDistanceToNearestEnemy(state1)
        currentEnemyDistance = self.getDistanceToNearestEnemy(state2)
        return prevEnemyDistance > currentEnemyDistance

    def gotCloserToFood(self, state1, state2):
        prevFoodDistance = self.getDistanceToNearestFood(state1)
        currentFoodDistance = self.getDistanceToNearestFood(state2)
        return prevFoodDistance > currentFoodDistance

    def gotCloserToDefendingFood(self, state1, state2):
        prevFoodDistance = self.getDistanceToNearestDefendingFood(state1)
        currentFoodDistance = self.getDistanceToNearestDefendingFood(state2)
        return prevFoodDistance > currentFoodDistance

    def depositedFood(self, state1, state2):
        return self.isPacman(state1) and self.isGhost(state2) and self.collectedFood > 0

    def pickedUpFood(self, state1, state2):
        prevAmountOfFood = len(self.getFood(state1).asList())
        currentAmountOfFood = len(self.getFood(state2).asList())
        return currentAmountOfFood < prevAmountOfFood

    def wentFurtherFromInitialState(self, state1, state2):
        prevDistance = self.getDistanceFromInitialState(state1)
        currentDistance = self.getDistanceFromInitialState(state2)
        return currentDistance > prevDistance

    def gotCloserToPowerCapsule(self, state1, state2):
        prevDistance = self.getDistanceToNearestPowerCapsule(state1)
        currentDistance = self.getDistanceToNearestPowerCapsule(state2)
        return prevDistance > currentDistance

    def isPacman(self, gameState):
        return gameState.getAgentState(self.index).isPacman

    def isGhost(self, gameState):
        return not self.isPacman(gameState)

    def isScared(self, gameState):
        return gameState.getAgentState(self.index).scaredTimer > 0

    def hasDied(self, state1, state2):
        stateOnePos = state1.getAgentState(self.index).getPosition()
        stateTwoPos = state2.getAgentState(self.index).getPosition()
        distance = self.distancer.getDistance(stateOnePos, stateTwoPos)
        return distance > 1

    def normalizeDistanceFeature(self, feature):
        result = float(feature) / self.maxMazeDistance
        return result

    ############################################################
    #  FEATURE EXTRACTORS THAT HAVEN'T BEEN USED ANYWHERE YET  #
    ############################################################

    # This is not implemented in getFeatures yet
    def getScoreForAmountOfFoodLeft(self, state):
        foodList = self.getFood(state).asList()
        return -len(foodList)

    # This is not implemented in getFeatures yet
    def getNumberOfSurroundingWalls(self, state):
        x, y = state.getAgentPosition(self.index)
        numberOfWalls = 0
        if state.hasWall(x + 1, y):
            numberOfWalls += 1
        if state.hasWall(x - 1, y):
            numberOfWalls += 1
        if state.hasWall(x, y + 1):
            numberOfWalls += 1
        if state.hasWall(x, y - 1):
            numberOfWalls += 1
        return numberOfWalls

    # This isn't implemented in getFeatures yet
    def willBeInCorridor(self, gameState, action):
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentPosition(self.index)
        legalActionsAtSuccessor = successor.getLegalActions(self.index)
        willBeInCorridor = len(legalActionsAtSuccessor) == 3 and action in legalActionsAtSuccessor and Directions.REVERSE(
            action) in legalActionsAtSuccessor
        return float(willBeInCorridor)


class OffensiveDummyAgent(DummyAgent):

    def getNewPositionFromTakingLegalMove(self, gameState, action):
        currentPos = gameState.getAgentPosition(self.index)
        if action is Directions.NORTH:
            return (currentPos[0], currentPos[1] + 1)
        elif action is Directions.SOUTH:
            return (currentPos[0], currentPos[1] - 1)
        elif action is Directions.EAST:
            return (currentPos[0] + 1, currentPos[1])
        elif action is Directions.WEST:
            return (currentPos[0] - 1, currentPos[1])
        elif action is Directions.STOP:
            return currentPos

    def getFeatures(self, gameState, action):
        """
        Calls superclass's getSharedFeatures and adds more features specific to the Offense Agent.
        """
        features = self.getSharedFeatures(gameState, action)

        successor = self.getSuccessor(gameState, action)

        # disable pellet-chasing features if

        # TODO: print out in the if block below: (could die on next enemy move)
        # we want to see if it's actually getting into the if block when it's supposed to be, i.e. when we're about to die
        #   - the successor position
        #   - the nearest enemy position
        # TODO: other stuff:
        #   - (DONE) apply the same shit we did for food pellets to the power capsules
        #   - apply the same shit we did for food pellets to the DEFENSIVE agent with the nearest enemy and whether we killed enemy, etc.

        successorPos = self.getNewPositionFromTakingLegalMove(gameState, action)

        nearestEnemyIndex = self.getNearestEnemyIndex(gameState) 
        opponentPos = gameState.getAgentPosition(nearestEnemyIndex) if (nearestEnemyIndex is not None) else None
        distanceToNearestEnemyFromSuccessor = self.distancer.getDistance(successorPos, opponentPos) if opponentPos else self.maxMazeDistance
        weAreKillable = gameState.getAgentState(self.index).isPacman and nearestEnemyIndex and not gameState.getAgentState(nearestEnemyIndex).isPacman
       
        if (weAreKillable): 
            print("WE COULD LITERALLY DIE")
            print("Potential New Pos: " + str(successorPos))
            print("Opponent Pos: " + str(opponentPos))

        if distanceToNearestEnemyFromSuccessor <= 1:
            print("CLOSE TO ENEMY!!!!!!!!!")
            print("Potential New Pos: " + str(successorPos))
            print("Opponent Pos: " + str(opponentPos))

        if self.isPoweredUp(successor):
            features['DistanceToNearestEnemy'] = 0.0

        if self.willDepositFoodAtSuccessor == 1.0:
            # features['WillDepositFoodAtSuccessor'] = self.willDepositFoodAtSuccessor(
                # gameState, successor)
            features['DistanceToNearestFood'] = 0.0
            # features['DistanceToNearestPowerCapsule'] = 0.0
            # features['StateHasFoodAtCurrentPosition'] = 0.0
            # features['StateHasPowerCapsuleAtCurrentPosition'] = 0.0
            print("in get features, features is now: ")


        distanceToNearestFood = self.getDistanceToNearestFood(successor) + 1.0
        # If there's food in the successor position right now, and there ISN'T food there anymore once we've taken the action:
        if gameState.hasFood(successorPos[0], successorPos[1]) and not self.stateHasFoodAtCurrentPosition(successor):
            distanceToNearestFood = 1.0
        features['DistanceToNearestFood'] = self.normalizeDistanceFeature(distanceToNearestFood) if distanceToNearestFood != 1.0 else 0.0 # THIS IS A BUG????????????
        
        distanceToNearestPowerCapsule = self.getDistanceToNearestPowerCapsule(successor) + 1.0
        if self.hasPowerCapsuleAtPosition(gameState, successorPos[0], successorPos[1]) and not self.hasPowerCapsuleAtPosition(successor, successorPos[0], successorPos[1]):
            distanceToNearestPowerCapsule = 1.0
        # features['DistanceToNearestPowerCapsule'] = self.normalizeDistanceFeature(distanceToNearestPowerCapsule) if distanceToNearestPowerCapsule != 1.0 else 0.0
        
        # features['StateHasFoodAtCurrentPosition'] = self.stateHasFoodAtCurrentPosition(successor)
        # features['StateHasPowerCapsuleAtCurrentPosition'] = self.hasPowerCapsuleAtPosition(successor, successorPos[0], successorPos[1])


        if self.justDied > 0:     
            # The enemy is really close to us and is also a ghost that could kill us (as opposed to a Pacman)
            print("SETTING FOOD FEATURES TO 0.0")
            print("self.index: " + str(self.index))
            print("Potential New Pos: " + str(successorPos))
            print("Opponent Pos: " + str(opponentPos))

            del features['DistanceToNearestFood']
            # del features['DistanceToNearestPowerCapsule']
            # del features['StateHasFoodAtCurrentPosition'] 
            # del features['StateHasPowerCapsuleAtCurrentPosition']
            self.justDied -= 1

        return features

    def getReward(self, prevState, action, currentState):
        """
        Calls superclass's getReward and adjusts reward value based on things specific to Offense Agent
        """
        reward = self.getSharedReward(prevState, action, currentState)

        if self.changedFromGhostToPacman(prevState, currentState):
            reward += 1

        if self.pickedUpFood(prevState, currentState):
            reward += 2
            self.collectedFood += 1

        # TODO maybe take out this elif, or potentially make it a separate reward condition. but probably dont need it at all
        elif self.gotCloserToFood(prevState, currentState):
            distanceToNearestFood = self.getDistanceToNearestFood(currentState)
            reward += 1 / distanceToNearestFood

        # distanceToNearestPowerCapsule = self.getDistanceToNearestPowerCapsule(
        #     currentState)
        # reward += 0.5 / distanceToNearestPowerCapsule

        if self.depositedFood(prevState, currentState):
            reward += self.collectedFood / self.totalFood
            self.collectedFood = 0.0

        if self.atePowerCapsule(prevState, currentState):
            reward += 2.5

        if self.isPacman(currentState):
            # print("i am pacm0n and i am running away from enemy ghosty bois")
            distanceToNearestEnemy = self.getDistanceToNearestEnemy(
                currentState) + 1.0
            reward -= 1.0 / distanceToNearestEnemy

        return reward

class DefensiveDummyAgent(DummyAgent):

    def getFeatures(self, gameState, action):
        features = self.getSharedFeatures(gameState, action)

        successor = self.getSuccessor(gameState, action)
        
        distanceToNearestDefendingFood = self.getDistanceToNearestDefendingFood(successor) + 1
        # features['DistanceToNearestDefendingFood'] = self.normalizeDistanceFeature(distanceToNearestDefendingFood) if distanceToNearestDefendingFood != 1.0 else 0.0
        
        # features['NumberOfInvaders'] = self.numberOfInvaders(
        #     gameState) / self.numberOfEnemies
    

        return features

    def getReward(self, prevState, action, currentState):
        reward = self.getSharedReward(prevState, action, currentState)

        if self.hasEnemyDied(prevState, currentState):
            reward += 5.0

        distanceToNearestEnemy = self.getDistanceToNearestEnemy(currentState) + 1
        reward += 1.0 / distanceToNearestEnemy

        if self.gotCloserToDefendingFood(prevState, currentState):
            distanceToNearestDefendingFood = self.getDistanceToNearestDefendingFood(currentState)
            reward += (1.0 / distanceToNearestDefendingFood) if distanceToNearestDefendingFood != 0 else 0

        if self.changedFromGhostToPacman(prevState, currentState):
            reward -= 1.0

        agentName = "Offensive Agent" if isinstance(self, OffensiveDummyAgent) else "Defensive Agent"
        showOutput = (DEBUG_OFFENSE_ONLY and agentName is "Offensive Agent") or (DEBUG_DEFENSE_ONLY and agentName is "Defensive Agent")
        if showOutput:
            print(agentName + " RECEIVED REWARD  " + str(reward))

        return reward


# TODO just add 1 to distance so that we dont have to have a separate feature for whether the state actually has food

# FEATURES FOR BOTH
# (DONE) distance from nearest enemy
# (DONE) distance from initial state
# (DONE) whether we have died
# (DONE) if we are a scared ghost
# (DONE) moving towards enemy
# (DONE) will change from pacman to ghost
# (DONE) will change from ghost to pacman
# TODO maybe add a feature for "desire" -- track how long we've been in a state

# REWARD FOR BOTH
# (DONE) living reward of -0.1
# (DONE) reward for moving away from initial state (more negative penalty the closer we are to initial)


# FEATURES FOR OFFENSIVE
# (DONE) how close we are to food
# (DONE) how close we are to power capsule
# (DONE) whether we might die soon/on the next move
#        - (DONE) if so, set features for food/power capsule to 0
# (DONE) whether or not we are roided up !!! (DONE)
#        - (DONE) if so, set feature for distance from ghosts to 0
# (DONE) whether we will deposit food on the next move

# REWARD FOR OFFENSIVE
# (DONE) eating food
# (DONE) getting closer to food if we didn't just eat food
# (DONE) eating power capsule
# (DONE) depositing food
# (DONE) Moving away from enemies (more negative reward the closer we are to an enemy)


# FEATURES FOR DEFENSE
# (DONE) Maybe number of enemy invaders? no stahp THOSE ARE HIS NIPPLESSs

# REWARD FOR DEFENSIVE
# (DONE) moving towards enemies (more positive reward the closer we are to enemy)
# (DONE) hasEnemyDied
# (DONE) Eating a pacman -- check if any enemies have died, and if so, +5 points
# (DONE) changed from ghost to pacman -- negative reward of -1. We want to discourage this

###################################################################

# TODO'S FOR WAY FAR IN THE FUTURE (if sanity allows):
# maybe we should get distance from both enemies rather than just nearest?
# (maybe: features for stop and reverse, like in Baseline Agent)
# (perhaps later, deal with walls, corners, and corridors)
# Feature for Noisy Distance from both?? enemy agents if not within range
