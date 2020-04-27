# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # original: 0.9 (unchanged)
    answerDiscount = 0.9
    # CHANGED - original: 0.2
    answerNoise = 0.0
    return answerDiscount, answerNoise

# Prefer the close exit (+1), risking the cliff (-10)
def question3a():
    answerDiscount = 0.1
    answerNoise = 0.0
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward    

# Prefer the close exit (+1), but avoiding the cliff (-10)
def question3b():
    answerDiscount = 0.67 # increase discount so it can see far enough into future to take the roundabout way
    answerNoise = 0.5 # noise is large; introduces risk of falling off cliff, so prefer roundabout path
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward

# Prefer the distant exit (+10), risking the cliff (-10)
def question3c():
    answerDiscount = 0.8 # high discount means we're forward-thinking and want to get to the higher terminal
    answerNoise = 0.0 # noise of 0 means there's no chance of falling off the cliff, so shorter path is preferable
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward

# Prefer the distant exit (+10), avoiding the cliff (-10)
def question3d():
    answerDiscount = 0.8
    answerNoise = 0.5 # noise is large; introduces risk of falling off cliff, so prefer roundabout path
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward

# Avoid both exits and the cliff (so an episode should never terminate)
def question3e():
    answerDiscount = 0.1
    answerNoise = 1 # noise of 1 means that the agent will never end up where it wants to go, no matter what the other two values are
    answerLivingReward = -5 
    # alternatively, make the living reward higher than any of the terminal rewards
    return answerDiscount, answerNoise, answerLivingReward   


def question6():
    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
