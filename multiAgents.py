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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # Start from the game's own score: it already accounts for eating food,
        # winning, dying, and the per-step time penalty.
        score = successorGameState.getScore()

        foodList = newFood.asList()

        # Reward being close to the nearest pellet (reciprocal so nearby food
        # matters much more than distant food, and the term stays bounded).
        if foodList:
            nearestFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
            score += 10.0 / nearestFoodDist

        # Ghost handling: flee ghosts that can hurt us, chase ones we can eat.
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            if scaredTime > 0:
                # Edible ghost: getting closer is good (points on capture).
                if dist > 0:
                    score += 200.0 / dist
            else:
                # Dangerous ghost: heavily penalize being adjacent, ignore when far.
                if dist <= 1:
                    score -= 500.0

        # Discourage sitting still; STOP wastes a move and encourages thrashing.
        if action == Directions.STOP:
            score -= 5.0

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        # This is Pacman's turn (agent 0). Try every legal move, score the
        # resulting state with minimax, and keep the move with the best value.
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            # Pacman has just moved, so the recursion starts at the first ghost.
            value = self.minimax(gameState.generateSuccessor(0, action), self.depth, 1)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
        
    def minimax(self, gameState: GameState, depth: int, agentIndex: int):
        # Base case: stop if the game is over or we've searched deep enough,
        # and score the state with the evaluation function.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        numAgents = gameState.getNumAgents()

        # Agents take turns in order: Pacman (0), ghost 1, ghost 2, ... then back
        # to Pacman. The modulo wraps around to Pacman after the last ghost.
        nextAgentIndex = (agentIndex + 1) % numAgents

        # One "ply" of depth is a full round (Pacman + every ghost moving once),
        # so only drop a depth level once the turn wraps back to Pacman.
        if nextAgentIndex == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth

        # Branch on the current agent's legal moves, collecting each child's value.
        legalActions = gameState.getLegalActions(agentIndex)
        values = []

        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            values.append(self.minimax(successor, nextDepth, nextAgentIndex))

        # Pacman maximizes his score; every ghost minimizes it (worst case).
        if agentIndex == 0:
            return max(values)
        else:
            return min(values)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # alpha = best value MAX (Pacman) can guarantee so far; beta = best value
        # MIN (ghosts) can guarantee. They start fully open and tighten as we go.
        alpha = float('-inf')
        beta = float('inf')
        # Pacman's turn (agent 0): pick the move with the best alpha-beta value.
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            # Pacman has just moved, so recurse starting at the first ghost.
            value = self.alphaBeta(gameState.generateSuccessor(0, action), 1, self.depth, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            # Raise alpha as the root maximizer learns it can guarantee more.
            alpha = max(alpha, value)
        return bestAction
    
    def alphaBeta(self, gameState: GameState, agentIndex: int, depth: int, alpha: float, beta: float):
        # Base case: game over or depth exhausted -> score the state.
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        # Turn order wraps Pacman -> ghosts -> Pacman; one depth level is a full
        # round, so only decrement when the turn wraps back to Pacman.
        numAgents = gameState.getNumAgents()
        nextAgentIndex = (agentIndex + 1) % numAgents
        if nextAgentIndex == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth
        
        # Pacman is a maximizer node; every ghost is a minimizer node.
        if agentIndex == 0:
            return self.maxValue(gameState, agentIndex, nextAgentIndex, nextDepth, alpha, beta)
        else:
            return self.minValue(gameState, agentIndex, nextAgentIndex, nextDepth, alpha, beta)

    def maxValue(self, gameState: GameState, agentIndex: int, nextAgentIndex: int, nextDepth: int, alpha: float, beta: float):
        v = float('-inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.alphaBeta(successor, nextAgentIndex, nextDepth, alpha, beta))
            # Prune: if this exceeds beta, the minimizer above would never let us
            # reach here, so skip the rest. Strict '>' avoids pruning on equality.
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def minValue(self, gameState: GameState, agentIndex: int, nextAgentIndex: int, nextDepth: int, alpha: float, beta: float):
        v = float('inf')
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = min(v, self.alphaBeta(successor, nextAgentIndex, nextDepth, alpha, beta))
            # Prune: if this drops below alpha, the maximizer above already has a
            # better option, so skip the rest. Strict '<' avoids pruning on equality.
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            value = self.expectimax(gameState.generateSuccessor(0, action), 1, self.depth)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
    
    def expectimax(self, gameState: GameState, agentIndex: int, depth: int):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        
        numAgents = gameState.getNumAgents()
        nextAgentIndex = (agentIndex + 1) % numAgents
        if nextAgentIndex == 0:
            nextDepth = depth - 1
        else:
            nextDepth = depth
        
        if agentIndex == 0:
            bestValue = float('-inf')
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = self.expectimax(successor, nextAgentIndex, nextDepth)
                if value > bestValue:
                    bestValue = value
            return bestValue
        else:
            chanceValue = 0
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                value = self.expectimax(successor, nextAgentIndex, nextDepth)
                chanceValue += value
            return chanceValue / len(gameState.getLegalActions(agentIndex))

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Starts from the game score and adjusts it with a few features.
    Ghosts: chase scared ghosts (reward grows as they get closer) and flee
    dangerous ones (heavy penalty when within 2 tiles). Food: eating is the
    dominant incentive via a large per-pellet penalty, with a small penalty on
    the distance to the nearest pellet to pull Pacman toward food. Capsules:
    penalize each remaining capsule so Pacman clears them too.
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    position = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    foodList = currentGameState.getFood()
    foodCount = foodList.count()
    powerCapsules = currentGameState.getCapsules()
    powerCapsuleCount = len(powerCapsules)
    
    for ghostState, scaredTime in zip(ghostStates, scaredTimes):
            ghostPos = ghostState.getPosition()
            dist = manhattanDistance(position, ghostPos)
            if scaredTime > 0:
                # Edible ghost: reward being on it, and reward closing in while
                # there's still enough scared time left to reach it.
                if dist == 0:
                    score += 1000.0
                elif dist < scaredTime:
                    score += 500.0 / (dist + 1)
            else:
                # Dangerous ghost: huge penalty for sharing a tile, smaller
                # penalty when within 2 tiles, ignored when farther away.
                if dist == 0:
                    score -= 2000.0
                elif dist <= 2:
                    score -= 500.0 / (dist + 0.5)

    # Food: the per-pellet penalty dominates so eating is always worthwhile,
    # while the small distance penalty gently pulls Pacman toward the nearest
    # pellet (Manhattan, so pacman might thrash slightly around walls).
    if foodCount > 0:
        nearestFoodDist = min(manhattanDistance(position, food) for food in foodList.asList())
        score -= 0.5 * nearestFoodDist
        score -= 20.0 * foodCount

    # Penalize each remaining capsule so Pacman is motivated to eat them.
    if powerCapsuleCount > 0:
        score -= 25.0 * powerCapsuleCount
    return score

# Abbreviation
better = betterEvaluationFunction
