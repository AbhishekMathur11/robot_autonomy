import numpy as np
import heapq

def CheckCondition(state, condition):
    """
    Check if the condition is satisfied in the state
    """
    if (np.sum(np.multiply(state, condition))-np.sum(np.multiply(condition, condition)))==0:
        return True
    else:
        return False

def CheckVisited(state,vertices):
    """
    Check if the state is already visited
    """
    for i in range(len(vertices)):
        if np.linalg.norm(np.subtract(state,vertices[i]))==0:
            return True
    return False

def ComputeNextState(state, effect):
    """
    Compute the next state by applying the effect
    """
    newstate=np.add(state, effect)
    return newstate


def Heuristic(state, GoalIndicesOneStep, GoalIndicesTwoStep): 
    """
    Compute the heuristic value for the state
    """
    score=0
    for idx in GoalIndicesOneStep:
        if state[idx[0]][idx[1]]==-1:
            score+=1

    for idx in GoalIndicesTwoStep:
        if state[idx[0]][idx[1]]==-1 and state[idx[0]][-1]==-1:
            score+=2
        elif state[idx[0]][idx[1]]==-1 and state[idx[0]][-1]==1:
            score+=1	
            
    return score

def InitializePreconditionsAndEffects(nrObjects, nrPredicates, Predicates, Objects):
    # TODO: Fill in the preconditions and effects for incomplete actions
    ActionPre=[]
    ActionEff=[]
    ActionDesc=[]

    ### Move to hallway
    for i in range(1,5,1):
        Precond=np.zeros([nrObjects, nrPredicates])
        Precond[0][0]=-1 #Robot not in hallway
        Precond[0][i]=1  #Robot in i-th room

        Effect=np.zeros([nrObjects, nrPredicates])
        Effect[0][0]=2.  # Robot in the hallway
        Effect[0][i]=-2. # Robot not in the i-th room
    
        ActionPre.append(Precond)
        ActionEff.append(Effect)
        ActionDesc.append("Move to InHallway from "+Predicates[i])

    ### Move to room
    for i in range(1,5,1):
        Precond=np.zeros([nrObjects, nrPredicates])
        Precond[0][0]=1  # Robot in the hallway
        Precond[0][i]=-1 # Robot not in the ith room

        Effect=np.zeros([nrObjects, nrPredicates])
        Effect[0][0]=-2. # Robot not in the hallway
        Effect[0][i]=2.  # Robot in the ith room

        ActionPre.append(Precond)
        ActionEff.append(Effect)
        ActionDesc.append("Move to "+Predicates[i]+" from InHallway")

    ### Move to Pantry 
    Precond=np.zeros([nrObjects, nrPredicates])
    # TODO: Robot in the kitchen and Robot not in the pantry
    Precond[0][Predicates.index("InKitchen")] = 1
    Precond[0][Predicates.index("InPantry")] = -1 

    Effect=np.zeros([nrObjects, nrPredicates])
    # TODO: Move robot from the kitchen to the pantry (remove from kitchen and add to pantry)

    Effect[0][Predicates.index("InKitchen")] = -2
    Effect[0][Predicates.index("InPantry")] = 2

    ActionPre.append(Precond)
    ActionEff.append(Effect)
    ActionDesc.append("Move to Pantry from Kitchen")

    ### Move from Pantry 
    Precond=np.zeros([nrObjects, nrPredicates])
    # TODO: Robot not in the kitchen and Robot in the pantry
    Precond[0][Predicates.index("InKitchen")] = -1
    Precond[0][Predicates.index("InPantry")] = 1

    Effect=np.zeros([nrObjects, nrPredicates])
    # TODO: Move robot from the pantry to the kitchen (remove from pantry and add to kitchen)

    Effect[0][Predicates.index("InKitchen")] = 2
    Effect[0][Predicates.index("InPantry")] = -2
    
    ActionPre.append(Precond)
    ActionEff.append(Effect)
    ActionDesc.append("Move to Kitchen from Pantry")

    ###Cut fruit in kitchen
    for j in [1,2]:
        Precond=np.zeros([nrObjects, nrPredicates])
        # TODO: Robot in the kitchen, fruit in the kitchen, knife in the kitchen, fruit not chopped
        Precond[0][Predicates.index("InKitchen")]=1 
        Precond[j][Predicates.index("InKitchen")]=1 
        Precond[4][Predicates.index("InKitchen")]=1
        Precond[j][Predicates.index("Chopped")]= - 1

         

        
        Effect=np.zeros([nrObjects, nrPredicates])
        # TODO: Fruit is chopped
        Effect[j][Predicates.index("Chopped")]= 2

        ActionPre.append(Precond)
        ActionEff.append(Effect)
        ActionDesc.append("Cut "+Objects[j]+" in the kitchen")

    ###Pickup object
    for i in range(1,6,1):
        for j in range(1,5,1):
            Precond=np.zeros([nrObjects, nrPredicates])
            Precond[0][i]=1 #Robot in ith room
            Precond[j][i]=1 #Object j in ith room
            Precond[j][-1]=-1 #Object j not on robot

            Effect=np.zeros([nrObjects, nrPredicates])
            Effect[j][i]=-2 #Object j not in ith room
            Effect[j][-1]=2 # Object j on robot

            ActionPre.append(Precond)
            ActionEff.append(Effect)
            ActionDesc.append("Pick up "+Objects[j]+" from "+Predicates[i])
        
    ###Place object
    for i in range(1,6,1):
        for j in range(1,5,1):
            Precond=np.zeros([nrObjects, nrPredicates])
            Precond[0][i]=1 #Robot in ith room
            Precond[j][i]=-1 #Object j not in ith room
            Precond[j][-1]=1 #Object j on robot

            Effect=np.zeros([nrObjects, nrPredicates])
            Effect[j][i]=2.  #Object j in ith room
            Effect[j][-1]=-2 #Object j not on robot

            ActionPre.append(Precond)
            ActionEff.append(Effect)
            ActionDesc.append("Place "+Objects[j]+" at "+Predicates[i])
            
    return ActionPre, ActionEff, ActionDesc

def InitializeStateAndGoal(nrObjects, nrPredicates):
    InitialState=-1*np.ones([nrObjects, nrPredicates])
    InitialState[0][0]=1 # Robot is in the hallway
    InitialState[1][4]=1 # Strawberry is in the garden
    InitialState[2][5]=1 # Lemon is in the pantry
    InitialState[3][2]=1 # Paper is in the office
    InitialState[4][2]=1 # Knife is in the office
    
    GoalState=np.zeros([nrObjects, nrPredicates])
    GoalState[0][1]=1 # Robot is in the kitchen
    GoalState[1][1]=1 # Strawberry is in the kitchen
    GoalState[2][4]=1 # Lemon is in the Garden
    GoalState[1][6]=1 # Strawberry is chopped
    
    return InitialState, GoalState

np.random.seed(13)

########### Construct the Initial State and Goal State ###########
Predicates=['InHallway', 'InKitchen', 'InOffice', 'InLivingRoom', 'InGarden','InPantry','Chopped','OnRobot']
Objects=['Robot','Strawberry','Lemon', 'Paper', 'Knife'] 
nrPredicates=len(Predicates)
nrObjects=len(Objects)

ActionPre, ActionEff, ActionDesc = InitializePreconditionsAndEffects(nrObjects, nrPredicates, Predicates, Objects)
InitialState, GoalState = InitializeStateAndGoal(nrObjects, nrPredicates)

# For Heuristic
GoalIndicesOneStep=[[0,1],[1,6]]
GoalIndicesTwoStep=[[1,1],[2,4]]

########### Search ###########
use_heuristic = True # Setting to True for A* and False for Dijkstra
vertices=[] # List of Visited States
parent=[]
action=[]
cost2come=[]
pq = [] # Use heapq to implement priority queue

# Insert the initial state
heapq.heappush(pq, (0, 0)) # (cost, vertex_id)
vertices.append(InitialState)
parent.append(0)
action.append(-1)
cost2come.append(0)

FoundPath=False
while len(pq)>0:
    
    # TODO: Implement Search
    # Get the element with the minimum cost
    P = np.array([cost2come[q] for q in range(len(pq))])

    id = P.argmin()

    x = pq[id][1]

    del pq[id]
    
    # Check for Goal, use CheckCondition

    if CheckCondition(vertices[x], GoalState):
        FoundPath=True
        break

    for i in range(len(ActionPre)):
        if CheckCondition(vertices[x], ActionPre[i]):
            # Compute the next state
            newstate=ComputeNextState(vertices[x], ActionEff[i])
            
            # Check if the next state is not visited
            if not CheckVisited(newstate, vertices):
                # Compute the cost
                if i < 10:
                    cost=1
                else:
                    cost=2
                
                # Compute the heuristic value
                h = Heuristic(newstate, GoalIndicesOneStep, GoalIndicesTwoStep) if use_heuristic else 0
                
                # Append to the priority queue
                heapq.heappush(pq, (cost+h, len(vertices)))
                
                # Append to vertices, parent, action, cost2come
                vertices.append(newstate)
                parent.append(x)
                action.append(i)
                cost2come.append(cost+cost2come[x])

    # For all actions

        # Check if the action is applicable, UseCheckCondition
        
        # Get the next state, use ComputeNextState
        
        # If the next state is not visited, add it to the queue. Check visited using CheckVisited
        # NOTE: 
        #   1. Append accordingly to vertices, parent, action, cost2come, and pq
        #   2. Compute the cost without heuristic for Dijkstra, and with heuristic for A*
    

print("Path Found: ", FoundPath)

# Extract Plan
Plan=[]
if FoundPath:
    while not x==0:
        Plan.insert(0,action[x])
        x=parent[x]
        
    # Print Plan
    print("States Explored: ", len(vertices))
    print("Plan Length: ", len(Plan))
    print()
    print("Plan:")
    for i in range(len(Plan)):
        print(ActionDesc[Plan[i]])