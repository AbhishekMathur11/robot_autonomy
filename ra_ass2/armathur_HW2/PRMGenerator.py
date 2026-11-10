import Franka
import numpy as np
import random
import pickle
import RobotUtil as rt
import time


random.seed(13)

#Initialize robot object
mybot=Franka.FrankArm()

#Create environment obstacles - # these are blocks in the environment/scene (not part of robot) 
pointsObs=[]
axesObs=[]

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,0,1.0]),[1.3,1.4,0.1])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,-0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1, 0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[-0.5, 0, 0.475]),[0.1,1.2,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

# Central block ahead of the robot
envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.45, 0, 0.25]),[0.5,0.4,0.5])
pointsObs.append(envpoints), axesObs.append(envaxes)

prmVertices=[] # list of vertices
prmEdges=[] # adjacency list (undirected graph)
start = time.time()
thresh = 3.0
k_neighbors = 30
# TODO: Create PRM - generate collision-free vertices
# TODO: Fill in the following function using prmVertices and prmEdges to store the graph. 
# The code at the end saves the graph into a python pickle file.



def PRMGenerator():
    global prmVertices
    global prmEdges
    global pointsObs
    global axesObs
    
    
    
    pointsObs = np.array(pointsObs)
    axesObs = np.array(axesObs)
    num_vertices = 1800

      
    attempts = 0
    while len(prmVertices) < num_vertices and attempts < num_vertices * 10:
        attempts += 1
        
        
        q_random = mybot.SampleRobotConfig()
        
        
        if not mybot.DetectCollision(q_random, pointsObs, axesObs):
            prmVertices.append(q_random)
            prmEdges.append([])  
            
            if len(prmVertices) % 100 == 0:
                print(f"Generated {len(prmVertices)} vertices...")
    
    print(f"Found {len(prmVertices)} collision-free configurations")
    
    
    print("Connecting vertices with edges...")
    
    
    for i in range(len(prmVertices)):
        
        distances = []
        for j in range(len(prmVertices)):
            if i != j:
                
                dist = np.linalg.norm(np.array(prmVertices[i]) - np.array(prmVertices[j]))
                distances.append((j, dist))
        
        
        distances.sort(key=lambda x: x[1])
        nearest_neighbors = distances[:k_neighbors]
        
        
        for j, _ in nearest_neighbors:
            
            if j in prmEdges[i] or i in prmEdges[j]:
                continue
                
            
            if not mybot.DetectCollisionEdge(prmVertices[i], prmVertices[j], pointsObs, axesObs):
                
                prmEdges[i].append(j)
                prmEdges[j].append(i)
        
        if i % 100 == 0:
            print(f"Connected {i}/{len(prmVertices)} vertices")

        

    #Save the PRM such that it can be run by PRMQuery.py
    # f = open("MyPRM.p", 'wb')
    f = open("/home/abhishek/MRSD/Robot Autonomy/MyPRM.p", 'wb')
    pickle.dump(prmVertices, f)
    pickle.dump(prmEdges, f)
    pickle.dump(pointsObs, f)
    pickle.dump(axesObs, f)
    f.close

if __name__ == "__main__":

    # Call the PRM Generator function and generate a graph
    PRMGenerator()

    print("\n", "Vertices: ", len(prmVertices),", Time Taken: ", time.time()-start)