# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
from baselineTeam import DefensiveReflexAgent

from pprint import pprint

TRAINING = True

DEBUG_OFFENSE_ONLY = True
DEBUG_DEFENSE_ONLY = False

WEIGHTS_AND_EXPLORATIONS_FILE = "weights_cece.json"
PREV_GAME_DATA_FILE = "prev_game_data.json"
MAX_ALPHA = 0.15

EMPTY_WEIGHTS_AND_EXPLORATIONS = {
    "offenseExplorations": {},
    "defenseExplorations": {},
    "offenseWeights": {},
    "defenseWeights": {}
}
FEATURES = {
    "offense": [
        "DistanceToNearestFood",
        "DistanceToNearestEnemy",
        "Bias",
        "Safety"
    ],
    "defense": [
        "DistanceToNearestDefendingFood",
        "DistanceToNearestEnemy",
        "Bias",
        "Safety"
    ]
}

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first="OffensiveAgent", second="DefensiveReflexAgent"):
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
        Handles the initial setup of the agent to populate useful fields 
        (such as what team we"re on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)

        # Either read from weights file if exists, or initialize weights to a util.Counter() with all features set to 0.0
        # At the end of the game, will write weights to file and create the file if it didn"t already exist
        self.weights = self.initializeWeights() # initializes them to 0 at the moment
        pprint("-- self.weights --")
        pprint(self.weights)

        # Exploration counter and constant
        self.explorations = self.initializeExplorations()
        pprint("-- self.explorations --")
        pprint(self.explorations)
        self.explorationConstant = 0.5 if TRAINING else 0.0

        # OTHER LEARNING PARAMETERS
        self.discount = 0.95
        self.alpha = 0.1 
        # probability that the agent will choose randomly instead of optimally; only relevant during training
        self.epsilon = 0.05 if TRAINING else 0.0
        self.livingReward = 0.0

        # Updates some of the data for the agent based on previous game results
        if TRAINING and path.exists(WEIGHTS_AND_EXPLORATIONS_FILE):
            self.initializeDataFromPreviousGame()

        # MAINTAINING GAME INFORMATION
        self.lastAction = Directions.STOP
        self.prevState = None
        maxSafeWidth = (gameState.data.layout.width / 2) - 1
        height = gameState.data.layout.height
        # self.safePositions is this list: [(15, 1), (15, 2), (15, 4), (15, 5), (15, 7), (15, 8), (15, 11), (15, 12), (15, 13), (15, 14)]
        self.safePositions = [(maxSafeWidth, y) for y in range(height) if not gameState.hasWall(maxSafeWidth, y)]
        self.border = None
        # The value of maxMazeDistance should be 76; any two positions in this maze can be at most 76 moves apart
        self.maxMazeDistance = max(self.distancer._distances.values())
        self.justDied = 0

        # KEEPING TRACK OF FOOD
        self.collectedFood = 0
        self.totalFood = len(self.getFood(gameState).asList())
        
        

    def chooseAction(self, gameState):
        """
        Picks the best of all legal actions based on their estimated Q values, which are computed
        with a linear combination of the feature values and their weights.
        This is the function that is called at every turn; any other functions that should be called
        each turn should be called from here.
        """
        s_prime = gameState #self.getCurrentObservation()
        a = self.lastAction
        # s = self.getPreviousObservation() if len(self.observationHistory) > 1 else s_prime
        s = self.prevState

        if self.prevState:
            reward = self.getReward(s, a, s_prime) # sets self.justDied = 1 if we died on the last transition
            if TRAINING:
                # Update weights based on reward received from the move we just took
                self.updateWeights(s, a, s_prime, reward)

        # Choose our next action!
        actions = gameState.getLegalActions(self.index)

        qValuesOfNextActions = [self.evaluatePotentialNextState(gameState, a) for a in actions]
        maxValue = max(qValuesOfNextActions)
        # bestActions = []
        # for i in range(len(actions)):
        #     action, value = actions[i], qValuesOfNextActions[i]
        #     if value == maxValue:
        #         bestActions.append(action)
        bestActions = [a for a, v in zip(actions, qValuesOfNextActions) if v == maxValue]

        # If there are 2 (or fewer) pellets left, we can win the game, so the best action will be 
        # the one that moves us closer to where we initially started
        # foodLeft = len(self.getFood(gameState).asList())
        # if foodLeft <= 2:
        #     actionChoice = self.getActionToGoBackHome(gameState, actions)
        #     self.updateFields(gameState, actionChoice)
        #     return actionChoice

        # Do a coin flip with probability self.epsilon to choose randomly instead of optimally, only if TRAINING
        coin_flip = util.flipCoin(self.epsilon)
        if TRAINING and coin_flip:
            actionChoice = random.choice(actions)
            self.updateFields(gameState, actionChoice)
            return actionChoice

        # In all other cases, choose the best action based on computed Q values
        # If multiple actions are tied, break ties randomly.
        actionChoice = random.choice(bestActions)
        self.updateFields(gameState, actionChoice)
        return actionChoice

    def updateFields(self, gameState, actionChoice):
        if self.prevState and self.isPacman(self.prevState) and not self.isPacman(gameState):
            self.border = self.prevState.getAgentPosition(self.index)
            
        self.lastAction = actionChoice
        self.prevState = gameState
        currentPos = gameState.getAgentPosition(self.index)
        self.explorations[(currentPos, actionChoice)] = self.explorations[(currentPos, actionChoice)] + 1

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
    
    def evaluatePotentialNextState(self, gameState, action):
        return self.getOptimisticQValue(gameState, action)

    def getOptimisticQValue(self, gameState, action):
        vanillaQValue = self.getQValue(gameState, action)
        position = gameState.getAgentPosition(self.index)
        bias = self.explorationConstant / float(self.explorations[(position, action)] + 1.0)
        return vanillaQValue + bias

    def getQValue(self, gameState, action):
        """
        Returns a number representing Q(gameState,action) = linear combination of all feature values and their weights
        = weightsVector * featureVector, where * is the dotProduct operator
        Example: [0.1 0.2 0.5] dotProduct [f1_value f2_value f3_value]
        """
        # weights is a Counter of {"feature name 1": weight1, "feature name 2": weight2, ...}
        weights = self.getWeights()
        # features is a Counter of {"feature name 1": featureValue1, "feature name 2": featureValue2, ...}
        features = self.getFeatures(gameState, action)

        # multiplication of two Counters returns the dot product of their elements
        estimatedQValue = weights * features
        return estimatedQValue

    def getWeights(self):
        return self.weights

    def updateWeights(self, prevGameState, action, currentGameState, reward):
        """
        Updates self.weights based on the transition we JUST TOOK, i.e. the action that
        brought us from the previous state to the current state
        """
        # Difference =
        #    (Reward received just now + value of acting optimally from this state on)
        #    - (value of the state we were just in)
        
        featuresAndValues = self.getFeatures(prevGameState, action)
        print("UPDATING WEIGHTS WITH THESE FEATURES: ")
        pprint(featuresAndValues)

        difference = reward + (self.discount * self.computeValueFromQValues(currentGameState)) - self.getQValue(prevGameState, action)
        
        for featureName, currentWeight in self.weights.items():
            self.weights[featureName] = currentWeight + self.alpha * difference * featuresAndValues[featureName]

        showOutput = (DEBUG_OFFENSE_ONLY and self.isOffensiveAgent()) or (DEBUG_DEFENSE_ONLY and not self.isOffensiveAgent())
        if showOutput:
            print("-- WEIGHTS:")
            pprint(self.weights)
            print("-- FEATURES:")
            pprint(featuresAndValues)

    def getSuccessor(self, gameState, action):
        """
        Copy-pasted from Baseline Team.
        Finds the next successor which is a grid position (location tuple).
        The gameState that this method returns is the same as the current one
        EXCEPT that the agent would have taken the specified action.
        We use this method for considering potential actions before choosing one.
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != util.nearestPoint(pos):
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
        if len(actions) == 0.0:
            return (None, 0.0)
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

    def final(self, gameState):
        """
        This is called at the end of the game. Overrides Capture Agent"s final method.
        In order to avoid bugs, just copy pasted what was in Capture Agent"s method
        instead of calling super().
        """
        print("### END OF GAME ###")

        if TRAINING and self.isOffensiveAgent():
            self.writeWeightsAndExplorationsToFile()
            self.writeCurrentGameDataToFile()

        # Copied from Capture Agent"s final method
        self.observationHistory = []

    ###########################
    # HANDING FILE READ/WRITE #
    ###########################

    def readWeightsAndExplorationsFromFile(self):
        if not path.exists(WEIGHTS_AND_EXPLORATIONS_FILE):
            raise Exception("The file {0} does not exist".format(WEIGHTS_AND_EXPLORATIONS_FILE))
        f = open(WEIGHTS_AND_EXPLORATIONS_FILE, "r")
        fileDict = json.load(f)
        f.close()
        return self.byteify(fileDict)

    def writeWeightsAndExplorationsToFile(self):
        pprint("Writing weights and explorations")
        weightsAndExplorations = self.readWeightsAndExplorationsFromFile() if path.exists(
            WEIGHTS_AND_EXPLORATIONS_FILE) else deepcopy(EMPTY_WEIGHTS_AND_EXPLORATIONS)
        pprint("-- weightsAndExplorations --")
        pprint(weightsAndExplorations)
        weightsAndExplorations["offenseWeights" if self.isOffensiveAgent() else "defenseWeights"] = self.weights
        explorationsWithStrings = {str(k):v for k,v in self.explorations.items()}
        weightsAndExplorations["offenseExplorations" if self.isOffensiveAgent() else "defenseExplorations"] = explorationsWithStrings
        f = open(WEIGHTS_AND_EXPLORATIONS_FILE, "w")
        json.dump(weightsAndExplorations, f)
        f.close()

    def generateDefaultWeights(self):
        weights = util.Counter()
        for feature in FEATURES["offense" if self.isOffensiveAgent() else "defense"]:
            weights[feature] = 0.0
        return weights

    def convertDictToCounter(self, d):
        result = util.Counter()
        for key, value in d.items():
            result[key] = value
        return result
    
    def convertExplorationsStringToTuple(self, s):
        for c in ["(", ")", " ", "\'"]:
            s = s.replace(c, "")
        sSplit = s.split(",")
        return ((int(sSplit[0]), int(sSplit[1])), sSplit[2])

    def byteify(self, item):
        if isinstance(item, dict):
            return {self.byteify(key): self.byteify(value)
                    for key, value in item.items()}
        elif isinstance(item, unicode):
            return item.encode("utf-8")
        else:
            return item

    def initializeWeights(self):
        if path.exists(WEIGHTS_AND_EXPLORATIONS_FILE):
            weightsDict = self.readWeightsAndExplorationsFromFile()["offenseWeights" if self.isOffensiveAgent() else "defenseWeights"]
            weightsCounter = self.convertDictToCounter(weightsDict)
            return weightsCounter
        else:
            return self.generateDefaultWeights()
    
    def initializeExplorations(self):
        if path.exists(WEIGHTS_AND_EXPLORATIONS_FILE):
            explorationsDict = self.readWeightsAndExplorationsFromFile()["offenseExplorations" if self.isOffensiveAgent() else "defenseExplorations"]
            explorationsDictWithTuples = {self.convertExplorationsStringToTuple(k):v for k,v in explorationsDict.items()}
            explorationsCounter = self.convertDictToCounter(explorationsDictWithTuples)
            return explorationsCounter
        else:
            return util.Counter()
    
    def readPreviousGameDataFromFile(self):
        if not path.exists(PREV_GAME_DATA_FILE):
            raise Exception("{0} does not exist".format(PREV_GAME_DATA_FILE))
        f = open(PREV_GAME_DATA_FILE, "r")
        prevGameData = json.load(f)
        f.close()
        return self.byteify(prevGameData)

    # The `winOrTie` property assumes we are the red team. Since we are 
    # only using this during training, this assumption is fine 
    def writeCurrentGameDataToFile(self):
        gameData = {
            "alpha": self.alpha,
            "winOrTie": self.getScore(self.getCurrentObservation()) >= 0.0
        }
        f = open(PREV_GAME_DATA_FILE, "w")
        json.dump(gameData, f)
        f.close()

    def initializeDataFromPreviousGame(self):
        if not path.exists(PREV_GAME_DATA_FILE):
            return 
        previousGameData = self.readPreviousGameDataFromFile()
        # if previousGameData["winOrTie"]:
        #     self.alpha = previousGameData["alpha"] * 0.8
        # else:
        #     self.alpha = min(previousGameData["alpha"] * 1.1, MAX_ALPHA)
            
    ####################
    # HELPER FUNCTIONS #
    ####################
    
    def isOffensiveAgent(self):
        return isinstance(self, OffensiveAgent)

    def getNearestEnemyIndex(self, gameState):
        # print("FINDING NEAREST ENEMY INDEX")
        currentPos = gameState.getAgentPosition(self.index)
        # print(" -- my pos: " + str(currentPos))
        opponentIndices = self.getOpponents(gameState) # returns a list of all enemies' agent indexes, i.e. [1, 3, 5]
        minDistance = self.maxMazeDistance
        nearestEnemyIndex = None
        for opponentIndex in opponentIndices:
            if gameState.getAgentPosition(opponentIndex) == None:
                continue
            opponentPos = gameState.getAgentPosition(opponentIndex)
            # print(" -- current enemy (index " + str(opponentIndex) + ") pos: " + str(opponentPos))
            distance = self.distancer.getDistance(currentPos, opponentPos)
            # print(" -- distance from this enemy: " + str(distance))
            if distance < minDistance:
                minDistance = distance
                nearestEnemyIndex = opponentIndex
            
        return nearestEnemyIndex

    def getDistanceToNearestEnemy(self, state):
        currentPos = state.getAgentState(self.index).getPosition()
        nearestEnemyIndex = self.getNearestEnemyIndex(state)
        opponentPos = state.getAgentPosition(nearestEnemyIndex) if (nearestEnemyIndex is not None) else None
        distance = self.distancer.getDistance(currentPos, opponentPos) if opponentPos else self.maxMazeDistance
        return distance

    def getDistanceToNearestEnemyFromPosition(self, currentState, queryPos):
        nearestEnemyIndex = self.getNearestEnemyIndex(currentState)
        # print("CALLING getDistanceToNearestEnemyFromPosition:")
        # print(" -- currently at " + str(currentState.getAgentPosition(self.index)) + ", querying about position " + str(queryPos))
        # print(" -- nearestEnemyIndex is " + str(nearestEnemyIndex))

        if nearestEnemyIndex is None:
            return None

        nearestEnemyPos = currentState.getAgentPosition(nearestEnemyIndex)
        # print(" -- nearestEnemyPos is " + str(nearestEnemyPos))
        distance = self.distancer.getDistance(queryPos, nearestEnemyPos)
        # print(" -- returning nearest enemy distance as: " + str(distance))
        return distance

    def pickedUpFood(self, state1, state2):
        prevAmountOfFood = len(self.getFood(state1).asList())
        currentAmountOfFood = len(self.getFood(state2).asList())
        return currentAmountOfFood < prevAmountOfFood

    def getDistanceToNearestFood(self, successor):
        foodList = self.getFood(successor).asList()
        minDistance = self.maxMazeDistance
        if len(foodList) > 0.0:  # This should always be True, but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        return minDistance

    def getDistanceToNearestFoodFromPosition(self, currentState, queryPos):
        foodList = self.getFood(currentState).asList()
        minDistance = self.maxMazeDistance
        if len(foodList) > 0:  # This should always be True, but better safe than sorry
            minDistance = min([self.getMazeDistance(queryPos, foodPos) for foodPos in foodList])
        return minDistance # + 1?
    
    def gotCloserToFood(self, state1, state2):
        prevFoodDistance = self.getDistanceToNearestFood(state1)
        currentFoodDistance = self.getDistanceToNearestFood(state2)
        return prevFoodDistance > currentFoodDistance

    def depositedFood(self, state1, state2):
        return self.isPacman(state1) and self.isGhost(state2) and self.collectedFood > 0.0

    def hasEdibleFoodAtPosition(self, gameState, pos1, pos2):
        foodPositions = self.getFood(gameState)
        return foodPositions[pos1][pos2]

    def isPacman(self, gameState):
        return gameState.getAgentState(self.index).isPacman

    def isGhost(self, gameState):
        return not self.isPacman(gameState)
    
    def isScaredGhost(self, gameState, index):
        return gameState.getAgentState(index).scaredTimer > 0.0

    def hasDied(self, state1, state2):
        stateOnePos = state1.getAgentState(self.index).getPosition()
        stateTwoPos = state2.getAgentState(self.index).getPosition()
        distance = self.distancer.getDistance(stateOnePos, stateTwoPos)
        return distance > 1.0

    def getDistanceFromNearestSafeState(self, state):
        currentPos = state.getAgentState(self.index).getPosition()
        distancesToSafe = [self.distancer.getDistance(currentPos, safePos) for safePos in self.safePositions]
        return min(distancesToSafe)

class OffensiveAgent(QLearningAgent):
    """
    A QLearningAgent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        # successor = self.getSuccessor(gameState, action)
        successor = gameState.generateSuccessor(self.index, action)
        # successorPos = self.getNewPositionFromTakingLegalMove(gameState, action)
        successorPos = successor.getAgentPosition(self.index)

        """
        Distance to the nearest Enemy
        """
        distanceToNearestEnemy = self.getDistanceToNearestEnemyFromPosition(gameState, successorPos) 
        # We care about the enemy if we're Pacman and can be killed
        if self.isPacman(gameState): #or self.isScaredGhost(gameState, self.index):  
            # print("we are killable so we care about enemies")      
            nearestEnemyIndex = self.getNearestEnemyIndex(gameState)
            nearestEnemyIsScaredGhost = self.isScaredGhost(gameState, nearestEnemyIndex) if nearestEnemyIndex is not None else False
            # If we couldn"t see any enemies in the first place, the enemy is too far away, or the enemy is scared, we don't care
            if (distanceToNearestEnemy is None or distanceToNearestEnemy > 5 or nearestEnemyIsScaredGhost): 
                # print("enemy is too far away so we actually don't care, setting feature to 0.0")
                # print(" -- distanceToNearestEnemy: " + str(distanceToNearestEnemy))
                features["DistanceToNearestEnemy"] = 0.0
            # Otherwise, we saw an enemy and we do care
            else:    
                normalizedDistanceToNearestEnemy = (5.0 - float(distanceToNearestEnemy)) / 10.0
                # print("enemy is close so we care!!! setting enemy feature to normalized value: " + str(normalizedDistanceToNearestEnemy))
                features["DistanceToNearestEnemy"] = normalizedDistanceToNearestEnemy

        """
        Distance to the nearest Food (pellet)
        """
        # # We care about food if we did not just die 
        # if self.justDied == 0:
        #     # If the enemy is really close by we actually don't care jk
        #     if distanceToNearestEnemy < 5:
        #         features["DistanceToNearestFood"] = 0.0
        #     else:
        #         distanceToNearestFood = self.getDistanceToNearestFoodFromPosition(gameState, successorPos)
        #         normalizedDistanceToNearestFood = -float(distanceToNearestFood + 1.0) / 30.0 # self.maxMazeDistance 
        #         features["DistanceToNearestFood"] = normalizedDistanceToNearestFood
        # # If we did just die, we don't set the food feature so that its weights don't get negatively affected
        # else:
        #     features["DistanceToNearestFood"] = 0.0
        #     self.justDied -= 1.0

        if features['DistanceToNearestEnemy'] == 0.0:
            distanceToNearestFood = self.getDistanceToNearestFoodFromPosition(gameState, successorPos)
            normalizedDistanceToNearestFood = -float(distanceToNearestFood + 1.0) / 300.0 # self.maxMazeDistance 
            features["DistanceToNearestFood"] = normalizedDistanceToNearestFood
            

        """
        Safety feature: discourages us from holding too many pellets at once
        """
        safety = (float(self.collectedFood) * float(self.getDistanceFromNearestSafeState(successor))) / 200.0 #(self.totalFood * self.maxMazeDistance)
        features["Safety"] = safety

        """
        Bias term that helps stabilize convergence 
        (like b in y = mx + b, since we"re doing a linear combination)
        """
        features["Bias"] = 0.1

        return features

    def getReward(self, prevState, action, currentState):
        reward = self.livingReward

        # Punishment for Dying
        if self.hasDied(prevState, currentState):
            print("AH FUCK WE DIED")
            self.justDied = 1.0
            self.collectedFood = 0.0
            reward -= 500.0

        if self.depositedFood(prevState, currentState):
            self.collectedFood = 0.0

        # # Reward for Getting Closer to Food
        # distanceToNearestFood = self.getDistanceToNearestFood(currentState)
        # if self.gotCloserToFood(prevState, currentState):
        #     reward += 1.0 / distanceToNearestFood
        # # Punishment for Getting Further from Food
        # else:
        #     reward -= 1.0 / distanceToNearestFood
            
        # Reward for Picking Up Food
        if self.pickedUpFood(prevState, currentState):
            print("NOM NOM NOM!!!")
            if self.collectedFood == 0:
                reward += 10.0 
            else:
                reward -= self.collectedFood
            self.collectedFood += 1

        # # Reward for Dropping Off Food
        # if self.depositedFood(prevState, currentState):
        #     reward += self.collectedFood # / self.totalFood
        
        # # Reward for Avoiding Enemies
        # if self.isPacman(currentState):
        #     distanceToNearestEnemy = self.getDistanceToNearestEnemy(currentState) + 1.0
        #     reward -= 1.0 / distanceToNearestEnemy

        # # Punishment for holding too many pellets; does not affect reward if not carrying any food
        # reward -= float(self.collectedFood) / 4 # float(self.totalFood)

        return reward


class DefensiveAgent(QLearningAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we"re on defense (1) or offense (0)
        features["onDefense"] = 1.0
        if myState.isPacman:
            features["onDefense"] = 0.0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features["numInvaders"] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features["invaderDistance"] = min(dists)

        if action == Directions.STOP:
            features["stop"] = 1.0
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features["reverse"] = 1.0

        return features

    def getWeights(self, gameState, action):
        # Reward for going Near Pacman
        # Reward for Eating Pacman
        # Reward for being Near Food
        return {"numInvaders": -1000.0, "onDefense": 100.0, "invaderDistance": -10.0, "stop": -100.0, "reverse": -2.0}
