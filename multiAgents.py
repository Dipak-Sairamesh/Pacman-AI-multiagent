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


import random
import util
from game import Agent
from util import manhattanDistance

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.

    Commands to run :
    python pacman.py -p ReflexAgent -l testClassic
    python pacman.py --frameTime 0 -p ReflexAgent -k 1
    python pacman.py --frameTime 0 -p ReflexAgent -k 2
    python autograder.py -q q1
    python autograder.py -q q1 --no-graphics
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
        food = currentGameState.getFood() # Find out all food locations at the start
        food_list = food.asList() # Store food locations in the form of a list
        score = successorGameState.getScore()
        new_food_list = newFood.asList() # Store new food locations in the form of a list
        new_ghost_positions = successorGameState.getGhostPositions() # Get new ghost positions
        nearest_ghost = float('+Inf')
        nearest_food = float('+Inf')
        score_multiplier = 1.0 # Generating a score multiplier

        if newPos in food_list:
            score_multiplier += score_multiplier * 1.25
        # Storing all manhattan distances of food in a list
        manhattan_food_list = [manhattanDistance(newPos, food_positions) for food_positions in new_food_list]
        total_food_remaining = len(new_food_list) # Total food particles left to eat

        if total_food_remaining: # Check for nearest food when food is remaining
            nearest_food = min(manhattan_food_list)
        score += (10.0 / nearest_food) - total_food_remaining + score_multiplier

        for ghost_positions in new_ghost_positions: # Check for nearest ghost
            manhattan_ghost_list = manhattanDistance(newPos, ghost_positions)
            nearest_ghost = min([nearest_ghost, manhattan_ghost_list])

        if nearest_ghost < 3: # Taking away score if ghost is nearby
            score = score * 0.75

        return score

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
    Commands to run :
        python autograder.py -q q2
        python autograder.py -q q2 --no-graphics
        python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
        python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex): Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        gameState.generateSuccessor(agentIndex, action): Returns the successor game state after an agent takes an action
        gameState.getNumAgents(): Returns the total number of agents in the game
        gameState.isWin(): Returns whether or not the game state is a winning state
        gameState.isLose(): Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Defining max value for getting pacman's ideal directions (Maximizer)
        def max_value(gameState, depth):
            actions = gameState.getLegalActions(0) # Pacman moves
            if len(actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth: # Terminal states
                return self.evaluationFunction(gameState), None

            value = float('-Inf')
            new_action = None

            for action in actions:
                new_value = min_value(gameState.generateSuccessor(0, action), 1, depth)[0] # At index 1
                if new_value > value: # Updating values and actions
                    value = new_value
                    new_action = action
            return value, new_action

        # Defining min value for getting the worst directions of the pacman - towards ghosts (Minimizer)
        def min_value(gameState, index, depth):
            actions = gameState.getLegalActions(index)
            if len(actions) == 0: # Terminal state
                return self.evaluationFunction(gameState), None

            value = float('+Inf')
            new_action = None

            for action in actions:
                if index == gameState.getNumAgents() - 1: # Check for last pacman agent
                    new_value = max_value(gameState.generateSuccessor(index, action), depth + 1)[0]
                else:
                    new_value = min_value(gameState.generateSuccessor(index, action), index + 1, depth)[0]
                if new_value < value: # Updating values and actions
                    value = new_value
                    new_action = action
            return value, new_action

        return max_value(gameState, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        Commands to run:
        python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
        python autograder.py -q q3
        python autograder.py -q q3 --no-graphics
        """
        "*** YOUR CODE HERE ***"
        alpha = float('-Inf')
        beta = float('Inf')

        # Defining max value for getting pacman's ideal directions (Maximizer)

        def max_value(gameState, depth, alpha, beta):
            actions = gameState.getLegalActions(0) # Pacman's moves
            if len(actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth: # Terminal states
                return self.evaluationFunction(gameState), None

            value = float('-Inf')
            new_action = None

            for action in actions:
                new_value = min_value(gameState.generateSuccessor(0, action), 1, depth, alpha, beta)[0]
                if value < new_value: # v = max(v, successor values)
                    value = new_value # Updating values and actions
                    new_action = action
                if value > beta: # if v > B, return v
                    return value, action
                alpha = max(alpha, value) # Updating alpha with max value
            return value, new_action

        # Defining min value for getting the worst directions of the pacman - towards ghosts (Minimizer)

        def min_value(gameState, index, depth, alpha, beta):
            actions = gameState.getLegalActions(index)
            if len(actions) == 0: # Terminal states
                return self.evaluationFunction(gameState), None

            value = float('+Inf')
            new_action = None

            for action in actions:
                if index == gameState.getNumAgents() - 1: # Check for last pacman agent
                    new_value = max_value(gameState.generateSuccessor(index, action), depth + 1, alpha, beta)[0]
                else:
                    new_value = min_value(gameState.generateSuccessor(index, action), index + 1, depth, alpha, beta)[0]
                if new_value < value:
                    value = new_value # Updating values and actions
                    new_action = action
                if value < alpha: # if v < a, return v
                    return value, action
                beta = min(beta, value) # Updating beta with min value
            return value, new_action

        return max_value(gameState, 0, alpha, beta)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        Commands to run :
            python autograder.py -q q4
            python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
            python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10
            python pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10
        """
        "*** YOUR CODE HERE ***"

        def max_value(gameState, depth):
            actions = gameState.getLegalActions(0) # Pacman's moves
            if len(actions) == 0 or gameState.isWin() or gameState.isLose() or depth == self.depth: # Terminal states
                return (self.evaluationFunction(gameState), None)

            value = float('-Inf')
            new_action = None

            for action in actions:
                new_value = expectimax_value(gameState.generateSuccessor(0, action), 1, depth)[0]
                if value < new_value: # v = max(v, successor values)
                    value = new_value # Updating values and actions
                    new_action = action

            return value, new_action

        def expectimax_value(gameState, index, depth):
            actions = gameState.getLegalActions(index)
            if len(actions) == 0:
                return self.evaluationFunction(gameState), None

            value = 0
            new_action = None

            for action in actions:
                if index == gameState.getNumAgents() - 1: # Check for last pacman index
                    new_value = max_value(gameState.generateSuccessor(index, action), depth + 1)[0]
                else:
                    new_value = expectimax_value(gameState.generateSuccessor(index, action), index + 1, depth)[0]
                # Expectimax optimizer doesn't play optimally like Minimax
                # Replacing Minimizer nodes with chance nodes (average of all values)
                chance_value = new_value / len(actions)
                value = value + chance_value

            return value, new_action

        return max_value(gameState, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    Commands to run:
        python autograder.py -q q5
        python autograder.py -q q5 --no-graphics
    """
    "*** YOUR CODE HERE ***"
    current_pos = currentGameState.getPacmanPosition() # Get Pacman positions
    ghosts = currentGameState.getGhostStates() # Get ghost states
    foods = currentGameState.getFood() # Get food locations
    capsules = currentGameState.getCapsules() # Get capsule locations

    food_list_dist = [] # List of manhattan distances to food
    ghost_list_dist = [] # List of manhattan distances to ghosts
    scared_ghost_list_dist = [] # List of manhattan distances to scared ghosts

    if currentGameState.isWin(): # Check for WIN/LOSE conditions
        return float('Inf')
    if currentGameState.isLose():
        return float('-Inf')

    for food in foods.asList():
        food_list_dist = food_list_dist + [manhattanDistance(food, current_pos)]
    if len(food_list_dist) > 0:
        min_food_dist = min(food_list_dist) # Populating minimum food distances
    else:
        min_food_dist = 0

    for ghost in ghosts:
        if ghost.scaredTimer == 0:
            ghost_list_dist = ghost_list_dist + [manhattanDistance(current_pos, ghost.getPosition())]
        elif ghost.scaredTimer > 0:
            scared_ghost_list_dist = scared_ghost_list_dist + [manhattanDistance(current_pos, ghost.getPosition())]

    min_scared_ghost_dist = 0
    min_ghost_dist = 1

    if len(ghost_list_dist) > 0:
        min_ghost_dist = min(ghost_list_dist)
    if len(scared_ghost_list_dist) > 0:
        min_scared_ghost_dist = min(scared_ghost_list_dist)

    # Score Multipliers for score
    score = scoreEvaluationFunction(currentGameState)
    # With respect to the nearest food
    score += min_food_dist * -1.5
    # With respect to the nearest ghost
    score += (-2.0 / min_ghost_dist)
    # With respect to the nearest scared ghost
    score += min_scared_ghost_dist * -0.85
    # With respect to number of capsules remaining
    score += len(capsules) * -15.0
    # With respect to total food remaining
    score += len(foods.asList()) * -4.0

    return score

# Abbreviation
better = betterEvaluationFunction
