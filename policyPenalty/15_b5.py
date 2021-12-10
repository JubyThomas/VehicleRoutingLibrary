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

# [START data_model]



def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] =np.array([
                                        [  0. ,  255.54, 560.61 ,390.91 ,685.05 ,361.45 ,543.84 ,590.68, 572.37, 483.89,
                                         353.57, 643.82, 483.14, 413.19 ,489.45],
                                         [255.54,   0.  , 473.83 ,232.54, 583.13 ,120.62, 300.06, 481.55 ,369.44 ,395.35,
                                        166.17, 398.25, 318.35 ,385.69, 451.69],
                                         [560.61, 473.83,   0.,   248.87, 124.91 ,406.34, 417.79 , 53.31, 278.77 , 80.28,
                                         317.91, 464.42, 199.12, 163.25 ,108.67],
                                         [390.91, 232.54, 248.87 ,  0. ,  351.5  ,158.49, 233.11, 250.22 ,188.18, 176.24,
                                          69.35, 318.64,  93.62, 214.48 ,256.53],
                                        [685.05 ,583.13 ,124.91 ,351.5  ,  0.  , 501.86, 473.22, 101.61, 318.  , 201.81,
                                         420.34, 496.15, 279.29, 285.36, 220.05],
                                        [361.45 ,120.62 ,406.34 ,158.49 ,501.86,   0.  , 182.99, 402.46 ,253.87, 334.72,
                                         93.01 ,282.7 , 224.15, 358.79 ,411.  ],
                                        [543.84 ,300.06, 417.79, 233.11, 473.22 ,182.99 ,  0. ,  390.87 ,157.5,  371.05,
                                         214.55 ,100.02 ,222.64, 441.17 ,466.93],
                                        [590.68 ,481.55,  53.31, 250.22, 101.61 ,402.46 ,390.87,   0.  , 243.26, 108.08,
                                        319.27, 428.34, 182.89, 208.61, 161.05],
                                        [572.37, 369.44, 278.77, 188.18 ,318.  , 253.87, 157.5,  243.26,   0. ,  251.61,
                                         220.05, 185.74 ,116.47 ,345.96 ,349.86],
                                        [483.89, 395.35,  80.28 ,176.24, 201.81, 334.72 ,371.05, 108.08 ,251.61  , 0.,
                                         243.76, 431.67 ,148.77, 110.86 , 98.47],
                                         [353.57, 166.17, 317.91,  69.35, 420.34,  93.01 ,214.55, 319.27, 220.05 ,243.76,
                                         0. ,  310.46, 152.4,  266.92, 318.  ],
                                        [643.82, 398.25 ,464.42, 318.64 ,496.15 ,282.7  ,100.02, 428.34, 185.74, 431.67,
                                        310.46,   0. ,  285.27, 515.23, 530.  ],
                                         [483.14, 318.35, 199.12 , 93.62 ,279.29, 224.15 ,222.64 ,182.89, 116.47, 148.77,
                                          152.4 , 285.27,   0.  , 231.95 ,246.01],
                                        [413.19, 385.69, 163.25 ,214.48 ,285.36 ,358.79, 441.17 ,208.61, 345.96 ,110.86,
                                         266.92, 515.23, 231.95 ,  0. ,   76.42],
                                        [489.45, 451.69, 108.67 ,256.53, 220.05, 411.  , 466.93, 161.05 ,349.86,  98.47,
                                        318. ,  530.,   246.01 , 76.42 ,  0.  ],
                                 ])

    data['demands'] =[0, 20, 12, 30, 48, 45, 18, 28, 45, 38, 16, 23, 10, 18, 21]
    data['num_vehicles'] = 3
    data['vehicle_capacities'] = [75,75,75]

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
    print(sum(mandatoryNodes))


 

    # To remove dropped nodes from dictionary
    removedOtherNodes=[]

    for key,value in dictSort.items():
        if value not in mandatoryNodes:
            drop_nodes.append(key)
            removedOtherNodes.append(key)
        else:
            mandatoryNodesById.append(key)    

    print("Node id Which must be dropped",removedOtherNodes)                
    print("Mandatory Nodes By Id",mandatoryNodesById)
    print("Dropped Nodes", drop_nodes)

    for x in removedOtherNodes:
        dictSort.pop(x)        
    
    print(dictSort)
    return data
    # [END data_model]


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    print(f'Objective: {assignment.ObjectiveValue()}')
    # Display dropped nodes.
    """ dropped_nodes = 'Dropped nodes:'
    for node in range(routing.Size()):
        if routing.IsStart(node) or routing.IsEnd(node):
            continue
        if assignment.Value(routing.NextVar(node)) == node:
            dropped_nodes += ' {}'.format(manager.IndexToNode(node))
            drop_nodes.append(manager.IndexToNode(node))
    print(dropped_nodes)"""

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