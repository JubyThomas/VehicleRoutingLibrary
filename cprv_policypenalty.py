import random
import math
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import matplotlib.pyplot as plt


coords=[]
noOfcities=15
data={}
dictSort={}
drop_nodes = []
mandatoryNodesById=[]


""" Method To form co-ordinates for the cities"""
def  createCityCordinates(noOfcities):
    for i in range(0,noOfcities):
        x = random.randint(0, 500)
        y = random.randint(0, 500)
        coords.append((int(x),int(y)) )
        #plt.scatter(x, y)

    #plt.show()
    print(coords)
    return coords

"""form co-ordinates for the depot some where far away"""
def formCoOrdinatesForDepot():    
    depotCoOrdinates=np.array([int(random.randint(0,750)),int(random.randint(0, 750))])
    print(depotCoOrdinates)
    return depotCoOrdinates


"""To form the the Demand"""
def createDemandForEachCity(noOfcities):      
    x=[]
    for i in  range (0,noOfcities+1):
        x.append(random.randint(10,40))
    data['demands']=x
    print(data['demands'])
    return data['demands']

def CreateDemandDictionarySorted(demandList):
    for val in range(len(demandList)):
        #print(val)
        dictSort[val]=demandList[val]
    print(dictSort)
    return dictSort

#Creating Array which contains details of mandatory nodes to be picked up
def createMandatoryVisitLocationList(demandList):
    y= sorted(demandList,reverse=True)
    print(y)
    mandatoryNodes=[]
    temp=0
    if(sum(y)>sum(data['demands'])):
     for testy in y:
        temp=sum(mandatoryNodes)+testy
        if(temp<=sum(data['demands'])):
           mandatoryNodes.append(testy)
    print("Mandatory Nodes To Be Picked")
    print(mandatoryNodes)
    return mandatoryNodes


# To remove dropped nodes from dictionary
def removedNodeList(mandatoryNodes):  
    removedOtherNodes=[]

    for key,value in dictSort.items():
        if value not in mandatoryNodes:
            drop_nodes.append(key)
            removedOtherNodes.append(key)
        else:
            mandatoryNodesById.append(key) 

    print("Node id Which must be dropped")                
    print(removedOtherNodes)

    for x in removedOtherNodes:
        dictSort.pop(x)      
    return dictSort    

def create_data_model():
    data={}
    
    coords=createCityCordinates(noOfcities)

    depotCordinate= formCoOrdinatesForDepot()
  
    """Adding the Depot Cordinates as first element in the List Of All City Coordinates"""
    coords.insert(0,depotCordinate)

    """ Creating Demands For each city"""
    x=[]
    for i in  range (0,noOfcities+1):
        x.append(random.randint(0,50))
    data['demands']=x
    print(data['demands'])
    CreateDemandDictionarySorted(x)

    """Call Function To Create Distance Matrix- creates Distance Matrix for all the co-ordinates , euclidean distance are calculated between two points"""
    data['distance_matrix']=np.zeros((noOfcities+1,noOfcities+1)) 
    for i in  range (0,noOfcities+1):
        for j in range(0,noOfcities+1):
            if (i==0 and j==0):
                data['distance_matrix'][i][j]=0
            else:
                data['distance_matrix'][i][j]=np.round(math.dist(coords[i],coords[j]),2)  
    print(data['distance_matrix'])

    mandatoryLocationList=createMandatoryVisitLocationList(x)

    removedList=removedNodeList(mandatoryLocationList)
    print(removedList)


    data['num_vehicles'] = 3
    data['vehicle_capacities'] = [50,50,50]    
    data['depot'] = 0
    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    print(f'Objective: {assignment.ObjectiveValue()}')
    # Display dropped nodes.
    dropped_nodes = 'Dropped nodes:'
    for node in range(routing.Size()):
        
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
    print(dropped_nodes)
    # Display routes
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(manager.IndexToNode(index),
                                                 route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total Distance of all routes: {}m'.format(total_distance))
    print('Total Load of all routes: {}'.format(total_load))

def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)


    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    # Allow to drop nodes.
    penalty = 10000
    for node in range(1, len(data['distance_matrix'])):       
        if(manager.NodeToIndex(node) in mandatoryNodesById):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
        else:
            routing.AddDisjunction([manager.NodeToIndex(node)], 0)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if assignment:
        print_solution(data, manager, routing, assignment)


if __name__ == '__main__':
    main()