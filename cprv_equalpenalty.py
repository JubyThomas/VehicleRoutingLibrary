import random
import math
import sys
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
#import matplotlib.pyplot as plt

coords=[]
dictSort={}
drop_nodes = []
mandatoryNodesById=[]
mandatoryNodes=[]
binSize=50 # we assume the bin size is 50 L for all our experiment
bin_fill_level=0.70
drop_nodes_greater_than70=[]

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


def CreateDemandDictionarySorted(demandList):
    for val in range(len(demandList)):
        #print(val)
        dictSort[val]=demandList[val]
    #print(dictSort)
    return dictSort

    #return data['distance_matrix']


def create_data_model(noOfcities,noOfVehicles,vehicleCapacity):
    data={}

    coords=createCityCordinates(noOfcities)

    depotCordinate= formCoOrdinatesForDepot()

    """Adding the Depot Cordinates as first element in the List Of All City Coordinates"""
    coords.insert(0,depotCordinate)

    """ Creating Demands For each city"""
    x=[]
    for i in  range (0,noOfcities):
        x.append(random.randint(0,50))
    data['demands']=x
    print(data['demands'])
    CreateDemandDictionarySorted(x)

    """Call Function To Create Distance Matrix"""
    data['distance_matrix']=np.zeros((noOfcities,noOfcities)) 
    for i in  range (0,noOfcities):
        for j in range(0,noOfcities):
            if (i==0 and j==0):
                data['distance_matrix'][i][j]=0
            else:
                data['distance_matrix'][i][j]=np.round(math.dist(coords[i],coords[j]),2)  
    np.set_printoptions(threshold=np.inf)
    print(data['distance_matrix'])

    y= sorted(data["demands"],reverse=True)
    print(y)
    
    data['num_vehicles'] = noOfVehicles
    
    data['vehicle_capacities']=np.zeros((noOfVehicles)) 
    for val in range(0,noOfVehicles):
         data['vehicle_capacities'][val] =  vehicleCapacity
   
    print("Number Of Vehicles:", data['num_vehicles'])  
    print("Vehicle Capacity of each vehicle:", data['vehicle_capacities'])     


    data['depot'] = 0
    return data


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    print(f'Objective: {assignment.ObjectiveValue()}')
    # Display dropped nodes.
    dropped_nodes = "Dropped nodes"
    for node in range(routing.Size()):

        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
            drop_nodes.append(manager.IndexToNode(node))
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
    print("*****************Result of Number of Nodes with Fill level >70% ****************************************")
    node_greaterthan_70=[]
    print("Total Number Of Demands:",len(data['demands']))               
    for val in range(0,len(data['demands'])):
        if(data['demands'][val] >= binSize*0.70 ):
            node_greaterthan_70.append(val)
            
    print("Dropped Nodes are:", drop_nodes)       
    for val in drop_nodes:
        if (val in node_greaterthan_70 ):
            drop_nodes_greater_than70.append(val)
   

    print("Nodes With fill Level greater than 70% :",node_greaterthan_70) 
    print("Total Number of Nodes With Fill level>70% :",len(node_greaterthan_70))
    print("Total Number of Dropped Nodes With Fill level>70% :",len(drop_nodes_greater_than70))
    print("Total Number of  Visited Nodes Nodes With Fill level>70% :",len(node_greaterthan_70)- len(drop_nodes_greater_than70))
    print("***********************************************************")

def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    noOfcities = int(input("Enter total number of locations including the depot: ")) 
    noOfVehicles = int(input("Enter total number of vehicles at the depot: ")) 
    vechileCapacity = int(input("Enter vehicle capacity of each vehicle: ")) 
    data = create_data_model(noOfcities,noOfVehicles,vechileCapacity)

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
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

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