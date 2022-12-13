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
from contest.game import Actions
from captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint
import distanceCalculator
from contest.util import PriorityQueue
from contest.util import Counter


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='aStarAttack', second='aStarDefend', num_training=0):
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

"""
we created an agent that will use a star algorithm to choose the action to take, 
eventhough a star computes the whole solution we will only take the first action 
and recompute the a star algorithm for every action since for different situations 
the problem (goal) for a star will be different.
Given the nature of this algorithm, sometimes the agents get stuck in some situations
"""
class aStarAgent(CaptureAgent):

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
        self.visited = Counter()
        self.stuck = 0
        self.mid_height = game_state.data.layout.height/2

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
    
    def evaluate(self, game_state, action):

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
        features = Counter()
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
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def nearFoodDist(self,game_state):
        #distance to nearest food
        food= self.get_food(game_state).as_list()
        my_pos = game_state.get_agent_state(self.index).get_position()

        if len(food)>0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food])
        else:
            min_distance = 0
        return min_distance

    """
        used to get some coordinates from our side of map, 
        to know where to get when we want to return back to defend
        or secure points
    """
    def home(self, game_state):

        #returns the list of positions in the boundary

        home = []
        middle_width = game_state.data.layout.width//2
        height = game_state.data.layout.height
        if self.red:
            i = middle_width-8
        else:
            i = middle_width +8
        home = [(i,j) for j in range(height)]
        finalHome = []
        for pos in home:
            if not game_state.has_wall(pos[0], pos[1]):
                finalHome.append(pos)
        return finalHome

    """
    returns the neares ghost distance and state.
    if we don't see any ghost returns None
    """
    def getGhostDistance(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position()!= None]
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

    """
    returns the minimum distance to an invader
    if we don't see one returns 10, a "high" distance
    """
    def getInvaderDistance(self, game_state):
        my_pos = game_state.get_agent_state(self.index).get_position()
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if  a.is_pacman and a.get_position()!= None]
        if len(invaders) > 0:
            minDist = min([self.get_maze_distance(my_pos, i.get_position()) for i in invaders])
        else:
            minDist = 10
        return minDist

    """
    aStar algorithm used to get the actions to take.
    it calculates the full path to the goal, but we will only 
    use the first action that it returns.
    It will be called for every action we want to take
    """

    def aStarSearch(self, game_state, problem, heuristic):

        start_state = problem.getStartState()

        frontier = PriorityQueue()
        h = heuristic(start_state, game_state)
        g = 0
        f = g + h
        start_node = (start_state, [], g)
        frontier.push(start_node, f)
        explored = []
        while not frontier.isEmpty():
            state,path,current_cost = frontier.pop()
            
            if state not in explored:
                explored.append(state)
                if problem.isGoalState(state):
                    return path
                successors = problem.getSuccessors(state)
                for successor in successors:
                    current_path = list(path)
                    successor_state = successor[0]
                    move = successor[1]
                    g = successor[2] + current_cost
                    h = heuristic(successor_state, game_state)
                    if successor_state not in explored:
                        current_path.append(move)
                        f = g + h
                        successor_node = (successor_state, current_path, g)
                        frontier.push(successor_node, f)
        return []


    """
    heuristic that takes into account the  distace to a ghost
    if it doesnt see a ghost heuristic = 0
    """
    def GhostHeuristic(self, state, game_state):
        heuristic = 0
        if self.getGhostDistance(game_state) != None:
            enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
            ghosts = [a for a in enemies if not a.is_pacman and a.scared_timer < 2 and a.get_position() != None]
            if ghosts != None and len(ghosts) > 0:
                ghostpositions = [g.get_position() for g in ghosts]
                ghostDists = [self.get_maze_distance(state,gp) for gp in ghostpositions]
                min_ghostDist = min(ghostDists)
                if min_ghostDist < 2:

                    heuristic = pow((5-min_ghostDist),5)

        return heuristic

    def foodHeuristic(self, state, game_state):
        return self.nearFoodDist(game_state)


"""
this agent will prioritize getting food, attacking. 
The first thing it will try is to get the capsule
"""
class aStarAttack(aStarAgent):

    def choose_action(self, game_state):
        print (game_state)
        my_state =  game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        nearestGhost = self.getGhostDistance(game_state)
        food = self.get_food(game_state).as_list()
        food_defending = self.get_food_you_are_defending(game_state).as_list()
        #home = self.distToHome(game_state)
        carry = my_state.num_carrying
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        nearestFood = self.nearFoodDist(game_state)
        capsule = self.get_capsules(game_state)
        dist_invader = self.getInvaderDistance(game_state)
        heuristic = self.GhostHeuristic

        """
        to avoid getting stuck we have a count of the times we
        have visited the same position.
        It sometimes gets stuck in the frontier of the two halves
        so when this happens we make it go further in its have of the map to
        try to unstuck it
        """
        if my_pos not in self.visited:
            self.visited[my_pos] = 0
        else:
            self.visited[my_pos] += 1

        if self.visited[my_pos] > 10:
            self.stuck =1

        if self.stuck ==1:
            problem = ReturnHome(game_state,self,self.index)
            actions = self.aStarSearch(game_state,problem,heuristic)
            if(len(actions) == 0):
                auxvisited = self.visited.copy()
                self.visited -=auxvisited
                #self.visited[stuck_position] = 0
                self.stuck = 0
                heuristic = self.foodHeuristic
                if self.red:
                    problem = GetDownFood(game_state,self,self.index)
                else:
                    problem = GetUpFood(game_state,self,self.index)
                actions = self.aStarSearch(game_state,problem,heuristic)
                if len(actions) == 0:
                    problem = SearchFood(game_state,self,self.index)
        
        else:
            #if wwe are very close to an invader and we are not scared, we go get it 
            if dist_invader < 5 and my_state.scared_timer == 0:
                problem = GetInvader(game_state,self,self.index)

            #check if there is still food to be eaten
            elif len(food) > 0:

                #if we are not safe, we are near a ghost we go back home
                if nearestGhost != None and nearestGhost[0] < 6 and nearestGhost[1].scared_timer < 5:
                    """
                    however, if there is still a capsule and we are closer to the capsule
                    than the ghost, we go get the capsule and afterwards we go eat food
                    """
                    if len(capsule) != 0:
                        if (self.get_maze_distance(my_pos,capsule[0])<nearestGhost[0]):
                            problem = GetCapsule(game_state,self,self.index)
                            actions = self.aStarSearch(game_state,problem,heuristic)
                            if len(actions) == 0:
                                heuristic = self.foodHeuristic
                                if self.red:
                                    problem = GetDownFood(game_state,self,self.index)
                                else:
                                    problem = GetUpFood(game_state,self,self.index)
                                actions = self.aStarSearch(game_state,problem,heuristic)
                                if len(actions)==0:
                                    problem = SearchFood(game_state,self,self.index)
                        
                        #if there is a capsule but the ghost is nearer
                        #than the capsule we go home
                        
                        else:
                            problem = ReturnHome(game_state, self, self.index)
                    
                   
                    #if the ghost is near and there is no capsule we return home
                    
                    else:
                        problem = ReturnHome(game_state, self, self.index)
                
                
                #if we don't see a ghost near, we go eat food, but first the capsule
                
                else:

                    
                    #we check for the scared timer of the ghosts
                    
                    if nearestGhost == None:
                        scared = -1
                    else:
                        scared = nearestGhost[1].scared_timer

                    """
                    we set a max_carry of food we can eat before going home depending on the situation.
                    once this max carry is reached we go home to secure the points
                    If we don't see a ghosts (scared = -1) or the ghost we see still has
                    a lot of scared time left and the nearest food is close, we are able to
                    carry more food since it is safer, then we will be able to secure more points at once
                    """
                    if (scared == -1 or scared > 5) and nearestFood < 5:
                        max_carry = 7
                    else:
                        max_carry = 2

                    """
                    we will return home when we have eaten enough food
                    and the nearest food is not one step away.
                    even if we reacehd the max_carry, if there is food one step away, we will eat it
                    """
                    if carry > max_carry and nearestFood > 1:
                        problem = ReturnHome(game_state, self, self.index)

                    else:
                        """
                        eat the capsule, and afterwards go get food
                        """
                        if len(capsule) != None:
                            problem = GetCapsule(game_state,self,self.index)
                            actions = self.aStarSearch(game_state,problem,heuristic)
                            if len(actions) == 0:
                                heuristic = self.foodHeuristic
                                if self.red:
                                    problem = GetDownFood(game_state,self,self.index)
                                else:
                                    problem = GetUpFood(game_state,self,self.index)
                                actions = self.aStarSearch(game_state,problem,heuristic)
                                if len(actions) == 0:
                                    problem = SearchFood(game_state,self,self.index)
                        else:
                            heuristic = self.foodHeuristic
                            if self.red:
                                problem = GetDownFood(game_state,self,self.index)
                            else:
                                problem = GetUpFood(game_state,self,self.index)
                            actions = self.aStarSearch(game_state,problem,heuristic)
                            if len(actions) == 0:
                                problem = SearchFood(game_state,self,self.index)
             
            
            #If there is no more food to be eaten we go back home
            
            else:
                problem = ReturnHome(game_state, self, self.index)
        
        actions = self.aStarSearch(game_state,problem,heuristic)
        if len(actions) == 0:
            return 'Stop'
        return actions[0]


"""
this agent will prioritize staying in defense, 
except in certain situations where it will also try to attack
"""
class aStarDefend(aStarAgent):
    def choose_action(self, game_state):
        my_state =  game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        nearestGhost = self.getGhostDistance(game_state)
        food = self.get_food(game_state).as_list()
        #home = self.distToHome(game_state)
        carry = my_state.num_carrying
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        heuristic = self.GhostHeuristic

        if my_pos not in self.visited:
            self.visited[my_pos] = 0
        else:
            self.visited[my_pos] += 1


        if self.visited[my_pos] > 10:
            self.stuck =1

        if self.stuck ==1:
            problem = ReturnHome(game_state,self,self.index)
            actions = self.aStarSearch(game_state,problem,heuristic)
            if(len(actions) == 0):
                heuristic = self.foodHeuristic
                auxvisited = self.visited.copy()
                self.visited -=auxvisited
                #self.visited[stuck_position] = 0
                self.stuck = 0
                problem = SearchFood(game_state,self,self.index)
        
        else:
            #if we are loosing, the "defender" will also atack. will grab only 2 food and go back
            if self.get_score(game_state) < -3:

                problem = SearchFood(game_state,self,self.index)
                heuristic = self.foodHeuristic
                actions = self.aStarSearch(game_state,problem,heuristic)
                heuristic = self.GhostHeuristic

                if len(actions) == 0 or carry > 1  or (nearestGhost != None and nearestGhost[0] < 3 and nearestGhost[1].scared_timer < 5):
                    problem = ReturnHome(game_state, self, self.index)
            else:    
                #if we win or tie, the defender will move around the food it is defending until it finds an invader
                if len(invaders) == 0:
                    
                    problem = SearchDefendingFood(game_state,self,self.index)
                else:
                    problem = GetInvader(game_state,self,self.index)
        
        actions = self.aStarSearch(game_state,problem,heuristic)
        if len(actions) == 0:
            return 'Stop'
        return actions[0]

    def get_features(self, game_state, action):
        features = Counter()
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
        self.walls = game_state.get-walls()
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


#the goal state is to be in a edible food
class SearchFood(SearchProblem):

    def __init__(self, game_state, agent,agentIndex):

        self.food = agent.get_food(game_state)

        self.walls = game_state.get_walls()
        self.start = game_state.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
    
    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.food.as_list()

#the goal state is to be in a defending food, to avoid getting stuck in the frontier
class SearchDefendingFood(SearchProblem):

    def __init__(self, game_state, agent,agentIndex):

        self.food = agent.get_food_you_are_defending(game_state)

        self.walls = game_state.get_walls()
        self.start = game_state.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
    
    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state == random.choice(self.food.as_list())

#the goal state is to be back ar your territory
class ReturnHome(SearchProblem):

    def __init__(self, game_state, agent,agentIndex):

        self.food = agent.get_food(game_state)
        self.capsule = agent.get_capsules(game_state)
        self.walls = game_state.get_walls()
        self.start = game_state.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
    
        self.home = agent.home(game_state)

    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.home

#the goal state is to be in a capsule
class GetCapsule(SearchProblem):

    def __init__(self, game_state, agent,agentIndex):

        self.food = agent.get_food(game_state)
        self.capsule = agent.get_capsules(game_state)
        self.walls = game_state.get_walls()
        self.start = game_state.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
    

    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.capsule

#the goal state is to be in the same position as an invader
class GetInvader(SearchProblem):

    def __init__(self, game_state, agent, agentIndex):

        self.food = agent.get_food(game_state)
        self.capsule = agent.get_capsules(game_state)
        self.walls = game_state.get_walls()
        self.start = game_state.get_agent_state(agentIndex).get_position()
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

class GoBack(SearchProblem):

    def __init__(self, game_state, agent,agentIndex):

        self.food = agent.get_food(game_state)
        self.capsule = agent.get_capsules(game_state)
        self.walls = game_state.get_walls()
        self.start = game_state.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
    

    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.capsule

class GetDownFood(SearchProblem):

    def __init__(self, game_state, agent,agentIndex):

        self.food = agent.get_food(game_state).as_list()

        self.walls = game_state.get_walls()
        self.start = game_state.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
        self.downFood =[]

        for f in self.food:
            if f[1] <  (agent.mid_height):
                self.downFood.append(f)

    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.downFood

class GetUpFood(SearchProblem):

    def __init__(self, game_state, agent,agentIndex):

        self.food = agent.get_food(game_state).as_list()

        self.walls = game_state.get_walls()
        self.start = game_state.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE
        self.downFood =[]

        for f in self.food:
            if f[1] >= (agent.mid_height):
                self.downFood.append(f)

    def isGoalState(self, state):
        #we reached the goal if we are in a capsule
        return state in self.downFood
