# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # initialize explored set to be empty
    explored = []
    # initialize frontier using initial state as a triple: start node, 
    # directions to get there (empty), and movement cost (0)
    frontier = util.Stack()
    frontier.push((problem.getStartState(), [], 0))

    while not frontier.isEmpty():
      # Remove top of stack for processing
      currentState, actions, cost = frontier.pop()

      # If we have explored the current node, return to top of loop
      if currentState in explored:
        continue
      
      # Add node to explored list
      explored += [currentState]

      # If we're in goal state return our list of actions
      if problem.isGoalState(currentState):
        return actions

      # Push successors to the stack for processing
      for state, action, cost in problem.getSuccessors(currentState):
        frontier.push((state, actions + [action], cost))

    # Return no solution if goal state not reached
    return []

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Initialize node (start state, action, path cost)
    node = (problem.getStartState(), [], 0)
    if problem.isGoalState(node[0]):
      return []
    
    # Initialize frontier as queue with node as only element
    frontier = util.Queue()
    frontier.push(node)

    # Initialize explored as empty list
    explored = []

    while not frontier.isEmpty():
      currentState, actions, cost = frontier.pop()

      if currentState in explored:
        continue

      explored += [currentState]

      if problem.isGoalState(currentState):
        return actions

      for state, action, cost in problem.getSuccessors(currentState):
        frontier.push((state, actions + [action], cost))
        
    # Return no solution if goal state is not reached
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Node initialized (start state, action, cost)
    node = (problem.getStartState(), [], 0)
    
    frontier = util.PriorityQueue()
    frontier.push(node, 0)

    explored = []

    while not frontier.isEmpty():
      currentState, actions, currentCost = frontier.pop()

      if problem.isGoalState(currentState):
        return actions

      if currentState in explored:
        continue

      explored += [currentState]
      
      for state, action, cost in problem.getSuccessors(currentState):
        if state not in explored:
          frontier.update((state, actions + [action], currentCost + cost), currentCost + cost)
    
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    node = (problem.getStartState(), [], 0 + heuristic(problem.getStartState(), problem))

    frontier = util.PriorityQueue()
    frontier.push(node,0)

    explored = []

    while not frontier.isEmpty():
      currentState, actions, currentCost = frontier.pop()

      if problem.isGoalState(currentState):
        return actions

      if currentState in explored:
        continue
 
      explored += [currentState]

      for state, action, cost in problem.getSuccessors(currentState):
        if state not in explored:
          frontier.update((state, actions + [action], currentCost + cost), currentCost + cost + heuristic(state, problem))

    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
