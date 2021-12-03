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

from numpy.lib.function_base import diff
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import random
import numpy as np
# [END import]


# [START data_model]



def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] =np.array([
                              [  0. ,  154.78, 140.13, 229.02 ,133.15 ,100.28 ,182.5,  158.47, 198.01, 196.21,
                                 184.62 ,320.05 , 59.67 , 93.05, 157.1 ],
                                [154.78,   0. ,  249.2,  101.12 ,204.4 , 206.4,  302.87, 167.07, 109.64, 87.28,
                                 261.82, 175.48 , 99.54, 124.65, 166.05],
                                [140.13, 249.2 ,   0.,   277.19, 266.6,  225.4 ,  54.57 ,298.61 ,325.98,240.69,
                                  60.53 ,424.62, 159.48, 232.13, 297.23],
                                [229.02, 101.12, 277.19 ,  0.  , 303.61, 300.33 ,331.1,  268.  , 197.33 , 36.72,
                                266.92, 191.43 ,169.5,  223.22 ,267.  ],
                                [133.15, 204.4,  266.6 , 303.61 ,  0.,    45.88, 299.11 , 69.35 ,167.26 ,280.72,
                                 316.41, 313. ,  161.62,  80.41 , 68.88],
                                [100.28, 206.4,  225.4  ,300.33 , 45.88  , 0.  , 255.05 ,106.79 ,193.38 ,273.21,
                                277.91, 336.79, 142.28,  85.63, 105.93],
                                [182.5,  302.87,  54.57, 331.1 , 299.11 ,255.05,   0. ,  339.09, 375.33, 294.72,
                                  85.76, 478.36, 210.54, 275.53 ,337.76],
                                [158.47, 167.07, 298.61, 268. ,   69.35, 106.79 ,339.09  , 0. ,  102.62, 252.04,
                                 340.2,  247.59, 161.4 ,  69.31  , 1.41],
                                [198.01 ,109.64, 325.98, 197.33, 167.26, 193.38, 375.33, 102.62,   0. ,  193.87,
                                 352.31 ,145.77, 167.04, 115.25 ,102.45],
                                [196.21,  87.28, 240.69  ,36.72, 280.72, 273.21, 294.72, 252.04 ,193.87,  0.,
                                 232.36 ,213.78, 137.35, 200.64, 250.92],
                                [184.62 ,261.82,  60.53 ,266.92 ,316.41, 277.91,  85.76, 340.2 , 352.31 ,232.36,
                                  0. ,  433.58, 187. ,  271.45 ,338.79],
                                [320.05 ,175.48, 424.62, 191.43 ,313.   ,336.79 ,478.36 ,247.59, 145.77, 213.78,
                                433.58 ,  0. ,  271.83 ,253.82, 247.55],
                                [ 59.67 , 99.54, 159.48 ,169.5 , 161.62 ,142.28 ,210.54, 161.4 ,167.04, 137.35,
                                187.,   271.83 ,  0. ,   92.96 ,160.  ],
                                [ 93.05 ,124.65, 232.13 ,223.22,  80.41 , 85.63, 275.53 , 69.31 ,115.25 ,200.64,
                                 271.45, 253.82, 92.96,   0. ,   67.9 ],
                                [157.1 , 166.05, 297.23, 267.  ,  68.88 ,105.93 ,337.76 ,  1.41 ,102.45, 250.92,
                                 338.79, 247.55, 160. ,   67.9 ,   0.  ],
                                 ])


    #"""To form the the Demand"""
    # if count ==1:
    data['demands'] =[0, 40, 12, 25, 19, 32, 38, 15, 34, 32, 16, 28, 18, 34, 12]

    data['num_vehicles'] = 3
    data['vehicle_capacities'] = [75,75,75]


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