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
from captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint
import distanceCalculator
from contest.util import PriorityQueue

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='DummyAgent', second='DummyAgent', num_training=0):
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
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class aStarAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def register_initial_state(self, game_state):
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
        CaptureAgent.register_initial_state in captureAgents.py.
        '''
        CaptureAgent.register_initial_state(self, game_state)
        self.start = game_state.get_agent_position(self.index)
        self.halfWidth = gameState.data.layout.width/2
        self.foodeaten = 0
        self.initFood = len(self.get_food(gameState).asList())
        self.lastFoodEatenPos = None
        self.initNumCapsules = len(self.get_capsules(gameState))

        '''
        Your initialization code goes here, if you need any.
        '''
    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

     def evaluate(self, games_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

     def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    def choose_action(self, game_state):
        self.LastEatenFoodPos(game_state)
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(action, values) if v == max_value]

        return random.choice(best_actions)

    def nearFoodDist(self,game_state):
        #distance to nearest food
        food_list = self.get_food(gameState).asList()
        my_pos = self.get_agent_state(self.index).get_position()

        if len(food)>0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
        else:
            min_distance = 0
        return min_distance

    def homeBoundaries(self, game_state):

        #returns the list of positions in the boundary

        boundaries = []
        middle_width = game_state.data.layout.width/2
        height = game_state.data.layout.height
        if self.red:
            i = middle_width-1
        else:
            i = middle_width +1
        boundaries = [(i,j) for j in range(height)]
        finalBoundaries = []
        for pos in boundaries:
            if not game_state.has_wall(pos[0], pos[1]):
                finalBoundaries.append(pos)
        return finalBoundaries

    def getGhostDistance(self, game_state):

        my_pos = self.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not is a.is_pacman and a.get_position()!= None]
        if len(ghosts) > 0:
            min_distance = 10000
            for g in ghosts:
                auxDist = self.get_maze_distance(my_pos, g.get_position())
                if auxDist < min_distance:
                    min_distance = auxDist
                    ghostState = g
            return [min_distance,ghostState]
        else:
            return None


    def aStarSearch(self, game_state, problem, heuristic):

        frontier = util.PriorityQueue() #frontier is now a priority queue so we can take into account the costs
        expandedNodes = list()
    
        start = [problem.getStartState(), [], 0]
        frontier.push(start,0)  #add to frontier the start node with priority 0
        while not frontier.isEmpty():
        
            node, path, cost = frontier.pop()
            if node not in expandedNodes:
                expandedNodes.append(node)  
        
                if problem.isGoalState(node) == True:
                    return path
    
                for child_node, child_path, child_cost in problem.getSuccessors(node):
                    if child_node not in expandedNodes:
                        new_path = path + [child_path]
                        g = cost + child_cost               #we calculate the priority with the cost accumulated in the node we are expanding + the cost of the child
                        h = heuristic(child_node, game_state)
                        f = g+h
                        frontier.push([child_node, new_path, g], f)         #we add the child in the frontier with a new cost that is the priority (so we accomulate), and the priority
        return []


    #heuristics
    def AvoidGhostHeuristic(self, state, game_state):
        nearestGhost = state.getGhostDistance(game_state)
        if nearestGhost != None:
            heuristic = -10 + nearestGhost[0]
    def EatOnlyFoodHeuristic(self, state, game_state):
        food = state.get_food().asList()
        pos = state.get_agent_state(state.index).get_position()
        foodDist = [state.get_maze_distance(pos,food) for food in food]
        minFoodDist = min(foodDist)
        heuristic = 0
        heuristic += 100/(len(food)+0.01)  
        heuristic += 1/(minFoodDist+0.01)
        return heuristic

    def EatFoodSafelyHeuristic(self, state, game_state):
        nearestGhost = state.getGhostDistance(game_state)
        if nearestGhost[0] < 5:
            heuristic = -10 + nearestGhost[0]

        food = state.get_food().asList()
        pos = state.get_agent_state(state.index).get_position()
        foodDist = [state.get_maze_distance(pos,food) for food in food]
        minFoodDist = min(foodDist)

        heuristic += 100/(len(food)+0.01)  
        heuristic += 1/(minFoodDist+0.01)
        return heuristic
   # def GetInvaderHeuristic(self, state, game_state)

class aStarAttack(aStarAgent):
    def choose_action(self, game_state):
        my_state =  game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        nearestGhost = self.getGhostDistance(game_state)
        if nearestGhost != None and nearestGhost[0] < 6 and nearestGhost[1].scared_timer < 5:
            problem = ReturnHome(game_state, self, self.index)
            return self.aStarSearch(game_state,problem,self.AvoidGhostHeuristic)[0]

        if nearestGhost != None and nearestGhost[1].scared_timer >=5:
            problem = SearchFood(game_state,self,self.index)
            return self.aStarSearch(game_state,problem,self.EatOnlyFoodHeuristic)[0]

        else:
            problem = SearchFood(game_state, self, self.index)
            return self.aStarSearch(game_state,problem,self.EatFoodSafelyHeuristic)[0]

class aStarDefend(aStarAgent):

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

        


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    def __init__(self, game_state, agentIndex):
        self.walls = gameState.getWalls()
        self.start = game_state.get_agent_state(self.index).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        return self.start

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

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors
    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class SearchCapsule(SearchProblem):

     def __init__(self, game_state, agent, agentIndex):

        self.food = agent.get_food(game_state)
        self.capsule = agent.get_capsules(game_state)
        self.walls = game_state.getWalls()
        self.start = agent.get_agent_state(self.index).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
    
    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.capsule

class SearchFood(SearchProblem):

     def __init__(self, game_state, agent,agentIndex):

        self.food = agent.get_food(game_state)

        self.walls = game_state.getWalls()
        self.start = game_state.get_agent_state(self.index).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
    
    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.food.asList()

class ReturnHome(SearchProblem):

     def __init__(self, game_state, agent,agentIndex):

        self.food = agent.get_food(game_state)
        self.capsule = agent.get_capsules(game_state)
        self.walls = game_state.getWalls()
        self.start = game_state.getAgentState(agentIndex).getPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
    
        self.home = self.homeBoundaries(game_state)

    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.home

class GetInvader(SearchProblem):

     def __init__(self, game_state, agent, agentIndex):

        self.food = agent.get_food(game_state)
        self.capsule = agent.get_capsules(game_state)
        self.walls = game_state.getWalls()
        self.start = game_state.geta_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
    
        self.enemies = [game_state.get_agent_state(i) for i in agent.get_opponents(game_state)]
        self.invaders = [a for a in self.enemies if a.is_pacman and a.get_position() is not None]
        if len(self.invaders) > 0:
            self.invadersPosition = [a.get_position() for a in self.invaders]
        else:
            self.invadersPosition = None
    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.invadersPosition
