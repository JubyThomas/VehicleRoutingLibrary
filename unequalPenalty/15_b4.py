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
                                    [  0. ,  280.61, 318.13, 194.79, 203.97,  69.78, 324.55, 216.45 ,156.32, 291.66,
                                    185.05 , 93.23, 266.02, 273.56, 431.82],
                                    [280.61 ,  0. ,  221.82, 194.83, 375.7 , 293.02, 367.64,  70.58, 170.66, 222.46,
                                     464.33, 188.15, 457.8 ,  60.21, 203.42],
                                    [318.13, 221.82,   0.  , 363.8 , 270.51, 371.65, 174.56 ,191.71, 316.99, 426.39,
                                    456.4 , 266.37, 341.06, 161.62 ,173.57],
                                    [194.79 ,194.83,363.8 ,   0.  , 379.7 , 158.46 ,450.92, 177.41 , 52.01 , 97.14,
                                    362.2,  134.06 ,452.29, 231.7 , 396.57],
                                     [203.97 ,375.7 , 270.51, 379.7  ,  0. ,  272.25, 167.3  ,306.67 ,30.83, 474.88,
                                    225.53 ,247.2 ,  82.1 , 335.72, 437.31],
                                    [ 69.78, 293.02 ,371.65 ,158.46, 272.25,  0. ,  393.01 ,238.82 ,136.69 ,249.97,
                                    204. ,  113.74, 329.71 ,300.04, 466.48],
                                    [324.55, 367.64 ,174.56, 450.92, 167.3  ,393.01 ,  0.  , 314.01 ,398.97, 534.24,
                                     391.97, 322.49 ,206.12, 311.28, 344.64],
                                    [216.45 , 70.58, 191.71, 177.41, 306.67 ,238.82, 314.01,   0.,   137.46, 234.8,
                                    397.77,127.28 ,388.67,  61.77, 230.99],
                                     [156.32, 170.66 ,316.99 , 52.01, 330.83, 136.69, 398.97, 137.46 ,  0.,   144.06,
                                     334.23 , 83.76 ,405.31, 196.47 ,365.25],
                                    [291.66, 222.46 ,426.39 , 97.14 ,474.88 ,249.97, 534.24 ,234.8,  144.06  , 0.,
                                     453.53, 227.79 ,548.73, 274.87 ,422.1 ],
                                    [185.05, 464.33, 456.4 , 362.2,  225.53 ,204. ,  391.97 ,397.77 ,334.23 ,453.53,
                                       0.,   278.21 ,228.02 ,451.05 ,597.61],
                                     [ 93.23 ,188.15, 266.37 ,134.06, 247.2 , 113.74, 322.49 ,127.28,  83.76, 227.79,
                                    278.21,   0.,   322.62, 187.45, 352.78],
                                     [266.02, 457.8,  341.06, 452.29,  82.1,  329.71, 206.12, 388.67 ,405.31, 548.73,
                                     228.02, 322.62,   0.   ,417.39 ,511.85],
                                    [273.56 , 60.21, 161.62, 231.7 , 335.72, 300.04 ,311.28,  61.77, 196.47, 274.87,
                                     451.05 ,187.45, 417.39 ,  0. ,  169.54],
                                    [431.82, 203.42, 173.57 ,396.57, 437.31 ,466.48 ,344.64, 230.99, 365.25, 422.1,
                                     597.61, 352.78, 511.85, 169.54,   0.  ],
                                 ])

    data['demands'] =[0, 15, 26, 31, 38, 12, 38, 18, 34, 25, 33, 31, 19, 14, 29]
    global total_demand_per_day
    total_demand_per_day=sum(data['demands'])
    global number_of_nodes
    number_of_nodes=len(data['demands'])
    
    data['num_vehicles'] = 3
    global number_of_routes_created
    number_of_routes_created=data['num_vehicles']
    
    
    data['vehicle_capacities'] = [75,75, 75]
    global effective_vehicle_capacity
    effective_vehicle_capacity=sum(data['vehicle_capacities'])

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
            
   
    print("Number Of nodes :",number_of_nodes) 
    print("Number of Routes Created:",number_of_routes_created)
    print("Number of Nodes Dropped:",len(drop_nodes))
    print("Total Demand Per Day :",total_demand_per_day)
    print("Unutilized Capacity :",effective_vehicle_capacity -total_load)
    print("Effective Vehicle Capacity :",effective_vehicle_capacity)
    print("\n")   
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