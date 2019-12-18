# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        "*** YOUR CODE HERE ***"
        
        # print("successorGameState: \n" + str(successorGameState))
        # print("score:" + str(successorGameState.getScore()))
        # print("newPos: " + str(newPos))
        # print("newFood: \n" + str(newFood))
        # print("newGhostStates: \n" + str(newGhostStates))
        # for state in newGhostStates:
        #     print(str(state.getPosition()) + "\n")
        # print("newScaredTimes: " + str(newScaredTimes) + "\n\n")
        
        "*** FOOD COUNT ***"
        currentFoodCount = currentGameState.getFood().count(True)
        newFoodCount = newFood.count(True)
        foodDiff = abs(currentFoodCount - newFoodCount)

        foodModifier = 0

        if foodDiff > 0:
            foodModifier = 20

        "*** FOOD DISTANCE ***"
        foodPositions = []
        for row in range(0,newFood.height):
            for col in range(newFood.width):
                if newFood[col][row]:
                    foodPositions.append((col,row))

        foodDistances = []
        for pos in foodPositions:
            foodDistances.append(manhattanDistance(newPos, pos))

        closestFood = 0

        if len(foodDistances) > 0:
            closestFood = min(foodDistances)

        "*** GHOST DISTANCE ***"
        ghostDistances = []
        for state in newGhostStates:
            ghostDistances.append(manhattanDistance(newPos, state.getPosition())) 

        closestGhost = min(ghostDistances)
        ghostDistanceModifier = 0

        scaredTime = min(newScaredTimes)

        if closestGhost < 2 and scaredTime == 0:
            ghostDistanceModifier = 20
        
        return successorGameState.getScore() - ghostDistanceModifier + foodModifier - closestFood

def manhattanDistance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def maxValue(gameState, depth):
            if terminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            v = float("-inf")
            actions = gameState.getLegalActions(0)

            for action in actions:
                nextState = gameState.generateSuccessor(0, action)
                v = max(v, minValue(nextState, 1, depth)) # Begin Ghosts
            return v

        def minValue(gameState, index, depth):
            if terminalState(gameState, depth):
                return self.evaluationFunction(gameState)
            
            v = float("inf")
            actions = gameState.getLegalActions(index)
            
            for action in actions:
                nextState = gameState.generateSuccessor(index, action)
                if index + 1 == gameState.getNumAgents():
                    v = min(v, maxValue(nextState, depth + 1)) # Pacman, down (up) a level 
                else:
                    v = min(v, minValue(nextState, index + 1, depth)) # Next ghost
            return v

        def terminalState(gameState, depth):
            # For some reason decrementing depth runs into a bug with expanded states
            # Starting and zero and expanding to self.depth resolves this.
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        bestValue = float("-inf")
        bestAction = None

        for action in gameState.getLegalActions():
            nextState = gameState.generateSuccessor(0, action)
            currentValue = minValue(nextState, 1, 0) # First ghost

            if currentValue > bestValue:
                bestValue = currentValue
                bestAction = action
        return bestAction
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def terminalState(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth
        
        def maxValue(gameState, alpha, beta, depth):
            if terminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            v = float("-inf")
            actions = gameState.getLegalActions(0)
            
            for action in actions:
                nextState = gameState.generateSuccessor(0, action)
                v = max(v, minValue(nextState, alpha, beta, 1, depth))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(gameState, alpha, beta, index, depth):
            if terminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            v = float("inf")
            actions = gameState.getLegalActions(index)

            for action in actions:
                nextState = gameState.generateSuccessor(index, action)
                if index + 1 == gameState.getNumAgents():
                    v = min(v, maxValue(nextState, alpha, beta, depth + 1)) # Back to Pacman
                else:
                    v = min(v, minValue(nextState, alpha, beta, index + 1, depth)) # Next Ghost
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        bestValue = float("-inf")
        bestAction = None
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions():
            nextState = gameState.generateSuccessor(0, action)
            currentValue = minValue(nextState, alpha, beta, 1, 0) # First ghost

            if currentValue > bestValue:
                bestValue = currentValue
                bestAction = action
            alpha = max(alpha, bestValue)
        return bestAction

            

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def terminalState(gameState, depth):
            return gameState.isWin() or gameState.isLose() or depth == self.depth

        def maxValue(gameState, depth):
            if terminalState(gameState, depth):
                return self.evaluationFunction(gameState)
            
            v = float("-inf")
            actions = gameState.getLegalActions(0)

            for action in actions:
                nextState = gameState.generateSuccessor(0, action)
                v = max(v, expectedValue(nextState, 1, depth))
            return v

        def expectedValue(gameState, index, depth):
            if terminalState(gameState, depth):
                return self.evaluationFunction(gameState)

            v = 0
            actions = gameState.getLegalActions(index)

            for action in actions:
                probability = 1 / len(actions)
                nextState = gameState.generateSuccessor(index, action)

                if index + 1 == gameState.getNumAgents():
                    v += probability * maxValue(nextState, depth + 1) # Pacman
                else:
                    v += probability * expectedValue(nextState, index + 1, depth) # Next ghost
            return v

        bestValue = float("-inf")
        bestAction = None
        
        for action in gameState.getLegalActions():
            nextState = gameState.generateSuccessor(0, action)
            currentValue = expectedValue(nextState, 1, 0) # First ghost

            if currentValue > bestValue:
                bestValue = currentValue
                bestAction = action
        return bestAction
        


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Modified my original reflex agent to account for current game state
    only.
    
    I'm taking into account the current score, the closest ghost (if the closest ghost
    if within a manhattan distance of 2 and not scared) and the distance to the closest
    food. I adjusted the weights slightly before my function maxed out the points.
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]

    "*** FOOD DISTANCE ***"
    foodPositions = []
    for row in range(0,currentFood.height):
        for col in range(currentFood.width):
            if currentFood[col][row]:
                foodPositions.append((col,row))

    foodDistances = []
    for pos in foodPositions:
        foodDistances.append(manhattanDistance(currentPos, pos))

    closestFood = 0

    if len(foodDistances) > 0:
        closestFood = min(foodDistances)

    "*** GHOST DISTANCE ***"
    ghostDistances = []
    for state in currentGhostStates:
        ghostDistances.append(manhattanDistance(currentPos, state.getPosition())) 

    closestGhost = min(ghostDistances)
    ghostDistanceModifier = 0

    scaredTime = min(currentScaredTimes)

    if closestGhost < 2 and scaredTime == 0:
        ghostDistanceModifier = 30

    return currentGameState.getScore() - ghostDistanceModifier - closestFood
    

# Abbreviation
better = betterEvaluationFunction
