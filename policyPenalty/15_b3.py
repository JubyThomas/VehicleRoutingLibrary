#!/usr/bin/env python3
# Copyright 2010-2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [START program]
"""Capacited Vehicles Routing Problem (CVRP)."""

# [START import]
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random
import numpy as np
# [END import]

dictSort={}
drop_nodes = []
mandatoryNodesById=[]
binSize=50 # we assume the bin size is 50 L for all our experiment
bin_fill_level=0.70
drop_nodes_greater_than70=[]

# [START data_model]



def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] =np.array([
                                [  0. ,  355.32, 368.15, 601. ,  300.27 ,706. ,  303.59, 704.23, 733.96, 648.57,
                                465.11, 403.07, 556.59 ,562.32, 573.77],
                                [355.32,   0. ,  208.95, 383.09,  90.47 ,404.03 , 86.03, 384.08 ,495.23, 308.08,
                                427.78 ,429.36, 219.54, 419.53 ,262.41],
                                 [368.15 ,208.95,   0. ,  234.7,  269.24, 350.27, 140.43, 361.56, 366.76, 337.83,
                                220.04 ,235.03, 256.69, 226.26, 238.  ],
                                [601.  , 383.09 ,234.7  ,  0.  , 466.86 ,190.14, 350.49 ,234.45, 133.18, 282.82,
                                258.79, 332.81, 261.77, 118.85 ,191.32],
                                [300.27,  90.47, 269.24 ,466.86  , 0. ,  494.39 ,128.86 ,473.21 ,583.03, 393.36,
                                 477.84, 464.56, 307.32 ,491.55, 352.82],
                                [706.,   404.03, 350.27, 190.14, 494.39,   0. ,  412.24 , 57.14, 169.19 ,150.76,
                                 444.29, 512. ,  197.27, 308.95 ,141.68],
                                [303.59,  86.03 ,140.43 ,350.49, 128.86 ,412.24 ,  0.  , 404. ,  474.38 ,345.03,
                                 351.64, 346.37, 253.05, 364.32, 274.31],
                                [704.23, 384.08 ,361.56, 234.45, 473.21,  57.14, 404.  ,   0.,   226.32, 101.32,
                                480.15, 542.9 , 167.03 ,352.53 ,130.46],
                                [733.96 ,495.23,366.76, 133.18 ,583.03, 169.19 ,474.38, 226.32  , 0. ,  312.16,
                                 367.14 ,447.83, 329.62 ,216.3,  257.66],
                                [648.57, 308.08, 337.83, 282.82, 393.36, 150.76, 345.03, 101.32, 312.16,   0.,
                                497.92, 548.64,  91.98 ,391.81, 107.42],
                                [465.11, 427.78, 220.04, 258.79 ,477.84 ,444.29 ,351.64, 480.15 ,367.14 ,497.92,
                                 0.  ,  84.01, 438.22, 150.84, 390.72],
                                [403.07, 429.36, 235.03, 332.81 ,464.56 ,512.  , 346.37, 542.9 , 447.83 ,548.64,
                                 84.01,   0.,   479.55 ,232.23, 441.58],
                                [556.59, 219.54, 256.69 ,261.77, 307.32, 197.27, 253.05, 167.03, 329.62,  91.98,
                                438.22 ,479.55 ,  0. ,  353.84 , 73.16],
                                [562.32, 419.53, 226.26, 118.85, 491.55, 308.95, 364.32 ,352.53, 216.3 , 391.81,
                                150.84, 232.23,353.84,   0.   ,290.9 ],
                                [573.77, 262.41 ,238. ,  191.32, 352.82 ,141.68, 274.31, 130.46, 257.66, 107.42,
                                390.72 ,441.58,  73.16 ,290.9 ,   0.  ],
                                 ])

    data['demands'] =[0, 15, 10, 12, 38, 32, 38, 27, 34, 25, 33, 23, 19, 37, 17]
    data['num_vehicles'] = 3
    data['vehicle_capacities'] = [75,75, 75]

    data['depot'] = 0

    dictSort={}

    for val in range(len(data['demands'])):
        dictSort[val]=data['demands'][val]
 
    print(dictSort)

    #Creating Array which contains details of mandatory nodes to be picked up
    y= sorted(data["demands"],reverse=True)
    print(y)
    mandatoryNodes=[]
    temp=0
    if(sum(y)>sum(data['vehicle_capacities'])):
     for testy in y:
        temp=sum(mandatoryNodes)+testy
        if(temp<=sum(data['vehicle_capacities'])):
           mandatoryNodes.append(testy)

    print("Mandatory Nodes To Be Picked")
    print(mandatoryNodes)
    

 

    # To remove dropped nodes from dictionary
    removedOtherNodes=[]

    for key,value in dictSort.items():
        if value not in mandatoryNodes:
            drop_nodes.append(key)
            removedOtherNodes.append(key)
        else:
            mandatoryNodesById.append(key)    

    for x in removedOtherNodes:
        dictSort.pop(x)     
    
    print(dictSort)
    return data
    # [END data_model]


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    print(f'Objective: {assignment.ObjectiveValue()}')
    # Display dropped nodes.
    dropped_nodes = []
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes.append(manager.IndexToNode(node))

    for x in dropped_nodes:
        if x in mandatoryNodesById:
            mandatoryNodesById.remove(x)
            drop_nodes.append(x) 

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
            
    
    for val in dropped_nodes:
        if (val in node_greaterthan_70 ):
            drop_nodes_greater_than70.append(val)
   
    print("Node id Which must be dropped",sorted(drop_nodes))          
    print("Mandatory Nodes By Id",mandatoryNodesById)
    print("Nodes With fill Level greater than 70% :",node_greaterthan_70) 
    print("Total Number of Nodes With Fill level>70% :",len(node_greaterthan_70))
    print("Total Number of Dropped Nodes With Fill level>70% :",len(drop_nodes_greater_than70))
    print("Total Number of  Visited Nodes Nodes With Fill level>70% :",len(node_greaterthan_70)- len(drop_nodes_greater_than70))
    print("***********************************************************")

def main():
    """Solve the CVRP problem."""
    # Instantiate the data problem.
   # for count in range(1,6):
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
     
    #Adding Penalty To Mandatory Nodes That Need To Be Picked Up

    penalty = 10000
    for node in range(1, len(data['distance_matrix'])):       
        if(manager.NodeToIndex(node) in [4,6,8,13]):
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