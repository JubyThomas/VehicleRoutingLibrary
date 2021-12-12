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
drop_nodes = []
binSize=50 # we assume the bin size is 50 L for all our experiment
bin_fill_level=0.70
drop_nodes_greater_than70=[]
# [END import]


# [START data_model]



def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] =np.array([
                           [  0. ,  536.59, 353.86 ,605.18 ,561.09, 417.81 ,499.12 ,360.9 , 386.94 ,602.63,
                            594.79, 219.49, 601.85, 255.79 ,367.36],
                            [536.59 ,  0. ,  263.06 ,194.24 ,413.09 ,167.72 , 94.25 ,332.05 ,232.26, 181.86,
                             367.03 ,317.65 , 78.29 ,292. ,  343.21],
                            [353.86, 263.06 ,  0. ,  257.12, 254.4 , 267.3 , 184.23,  76.66,  37.11 ,256.86,
                             260.31, 191.05 ,298.41 ,216.81 , 90.24],
                            [605.18 ,94.24 ,257.12 ,  0.  , 263.82, 339.61, 135.28 ,288.88 ,221. ,   13.04,
                            198.98 ,408.89, 147.01, 406.97, 293.16],
                            [561.09 ,413.09 ,254.4 , 263.82  , 0. ,  494.45 ,320.41, 207.18, 246.98 ,274.71,
                             75.72 ,442.56, 397.51, 471.21, 197.5 ],
                            [417.81 ,167.72, 267.3 , 339.61 ,494.45 ,  0.  , 206.23 ,343.43 ,256. ,  329.21,
                              469.95 ,207. ,  245.97 ,162.41, 357.26],
                            [499.12 , 94.25 ,184.23 ,135.28 ,320.41, 206.23  , 0.  , 246.09, 149.4,  126.21,
                            278.96, 289.07 ,114.24, 278.92, 256.07],
                            [360.9 , 332.05 , 76.66 ,288.88, 207.18, 343.43, 246.09 ,  0.,   100.12, 291.89,
                             233.89, 240.62, 357.86 ,276.38 , 14.  ],
                            [386.94 ,232.26,  37.11, 221.  , 246.98, 256. ,  149.4,  100.12,   0.,   220.33,
                             242.39, 211.46, 263.21, 229.73, 112.  ],
                            [602.63, 181.86, 256.86 , 13.04, 274.71 ,329.21, 126.21, 291.89 ,220.33,   0.,
                             210.92, 403.93 ,134.,   400.48 ,296.74],
                            [594.79, 367.03 ,260.31 ,198.98 , 75.72, 469.95, 278.96 ,233.89 ,242.39 ,210.92,
                               0. ,  451.13,340.46 ,472.12 ,227.95],
                             [219.49, 317.65 ,191.05, 408.89 ,442.56 ,207. ,  289.07, 240.62, 211.46 ,403.93,
                             451.13  , 0.,   385.29  ,52.8 , 253.15],
                            [601.85,  78.29 ,298.41 ,147.01,397.51 ,245.97 ,114.24, 357.86 ,263.21, 134.,
                            340.46 ,385.29 ,  0. ,  364.5  ,366.98],
                            [255.79, 292.  , 216.81, 406.97 ,471.21, 162.41 ,278.92, 276.38, 229.73, 400.48,
                            472.12,  52.8 , 364.5 ,   0. ,  289.69],
                             [367.36 ,343.21 , 90.24, 293.16, 197.5  ,357.26, 256.07 , 14. ,  112. ,  296.74,
                            227.95, 253.15 ,366.98, 289.69 ,  0.  ],
                           ])

    #"""To form the the Demand"""
    data['demands'] =[0, 17, 13, 19, 38, 40, 22, 11, 28, 37, 20, 10, 14, 33, 35]
    global total_demand_per_day
    total_demand_per_day=sum(data['demands'])
    global number_of_nodes
    number_of_nodes=len(data['demands'])
    
    data['num_vehicles'] = 3
    global number_of_routes_created
    number_of_routes_created=data['num_vehicles']
    
    data['vehicle_capacities'] = [75,75,75]
    
    global effective_vehicle_capacity
    effective_vehicle_capacity=sum(data['vehicle_capacities'])


    data['depot'] = 0
    return data
    # [END data_model]


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
            
    print("Number Of nodes :",number_of_nodes) 
    print("Number of Routes Created:",number_of_routes_created)
    print("Number of Nodes Dropped:",len(drop_nodes))
    print("Total Demand Per Day :",total_demand_per_day)
    print("Unutilized Capacity :",effective_vehicle_capacity -total_load)
    print("Effective Vehicle Capacity :",effective_vehicle_capacity)
    print("\n") 
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
    """
    penalty = 10000
    routing.AddDisjunction([manager.NodeToIndex(i) for i in [1,6,13]], penalty,4)"""

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