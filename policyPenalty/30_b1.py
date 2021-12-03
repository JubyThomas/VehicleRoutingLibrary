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
                                        [  0. ,  136.07 ,421.47, 374.82, 129.54 , 92.01, 312.7 , 568.84, 547.26 ,494.26,
                                         229.49 ,214.6,  364.68, 162.89 ,429.79, 185.61, 299.7,  451.09, 446.48 ,438.72,
                                         512.06, 580.02, 324.42, 386.65 ,490.18 ,351.59, 520.61, 240.26 ,339.87 ,522.01],
                                        [136.07,   0. ,  359.92 ,249.47, 141.78 , 46.62 ,249.93, 454.68, 437.91 ,359.,
                                        178.55,  79.01, 257.62, 119.27, 295.37 ,103.09, 164.56 ,330.15, 345.3 , 307.32,
                                         389.01, 466.  , 201.3,  250.6 , 363.93 ,269.14, 394.  , 165.6 , 323.02, 392.26],
                                        [421.47, 359.92,   0.  , 250.45, 497.29, 387.36, 111.32 ,261.35, 225.39, 460.93,
                                          192.92, 355.28, 169.96 ,478.6,  340.33, 453.2 , 321.73, 546.62 ,143.59 ,308.38,
                                          276.83, 268.36, 455.29, 366.9  ,287.63 ,102.84, 305.59, 194.59 ,137.36, 331.69],
                                        [374.82, 249.47 ,250.45 ,  0. ,  381.14, 295.1,  200.48 ,211.94 ,203.11, 213.66,
                                        219.49, 191.55 , 81.06 ,337.96 , 94.87, 297.14, 108.75 ,300.73 ,142.43 , 76.3,
                                      139.63, 223.08 ,231.95, 117.92, 115.38, 157.72, 145.8,  181. ,  319.6 , 149.4 ],
                                     [129.54 ,141.78, 497.29, 381.14,   0.,   110.22, 386.12, 591.05, 576.49, 442.04,
                                    308.25 ,190.69, 398.3,   57.43, 408.42, 100.96, 281.63, 364.27, 486.43, 428.6,
                                    519.94 ,602.34 ,250.22, 352.44, 491.9,  410.02, 520.99, 303.06, 442. ,  514.6 ],
                                    [ 92.01 , 46.62, 387.36, 295.1 , 110.22,   0.  , 276.1 , 498.25, 480.24, 402.36,
                                    198.13 ,123.18 ,298.5 , 106.89, 341.84 ,109.66, 210.94, 363.11 ,385.03 ,353.9,
                                    434.44 ,509.56 ,235.05 ,295.97, 409.91, 302.34 ,440.07, 193.6 , 334.86 ,438.73],
                                    [312.7  ,249.93 ,111.32 ,200.48, 386.12, 276.1 ,   0.,   301.48 ,270.42, 411.81,
                                    83.22, 253.58 ,127.44, 369.07 ,295.21 ,346.,   238.61, 471.56, 165.66, 273.31,
                                    282.51 ,311.19 ,366.55, 303.37 ,279.69,  58.31, 305.5 ,  84.38 ,119.28, 324.  ],
                                    [568.84, 454.68 ,261.35, 211.94 ,591.05 ,498.25 ,301.48,   0. ,   36.4 , 321.91,
                                    366.23 ,403.02 ,204.71 ,549.61 ,232.02 ,509.07, 319.51 ,459.68 ,135.82, 193.55,
                                    88.68  ,11.31 ,429.17 ,293.03 ,123.32 ,243.82 ,110.45 ,338.56 ,391.01, 140.07],
                                    [547.26 ,437.91 ,225.39 ,203.11, 576.49, 480.24, 270.42,  36.4,    0. ,  337.34,
                                    338.54 ,391.15 ,182.78, 537.51, 238.13, 498.38, 311.85, 469.11, 105.7  ,198.31,
                                    100.9   ,43.05 ,428.57, 295.85, 133.09, 213.6 , 128.16 ,313.02 ,356.24, 159.33],
                                    [494.26 ,359. ,460.93 ,213.66 ,442.04, 402.36, 411.81, 321.91, 337.34,   0.,
                                    413.73 ,280.   ,294.16 ,385.5 , 120.62, 341.08, 206.42, 151.4,  332.92, 154.87,
                                    236.52 ,329.11 ,196.35, 117.   ,204.97, 371.38 ,211.48, 374.33, 531. ,  182.33],
                                    [229.49 ,178.55 ,192.92, 219.49 ,308.25, 198.13 , 83.22, 366.23, 338.54, 413.73,
                                    0.   ,201.72 ,168.69, 297.06, 307.56, 279.81 ,216.81 ,446.2 , 233.06, 295.76,
                                    331.27 ,376.72 ,328.8  ,298.01, 320.56, 128. ,  349.38 , 39.56 ,147.83, 362.01],
                                    [214.6  , 79.01 ,355.28, 191.55, 190.69, 123.18 ,253.58, 403.02 ,391.15 ,280.,
                                    201.72  , 0.   ,221.38 ,146.6 , 221.96, 107.79,  91.92 ,260.23, 309.38, 238.8,
                                    329.51 ,414.24 ,133.54 ,173. ,  301.22, 255.2,  330.34, 174.24 ,349.01 ,324.78],
                                    [364.68 ,257.62 ,169.96 , 81.06, 398.3 , 298.5 , 127.44, 204.71 ,182.78, 294.16,
                                    168.69 ,221.38  , 0.   ,364.05, 174.16 ,328.69, 162.11, 377.77,  88.55 ,147.67,
                                    162.63 ,215.78 ,295.53 ,196.98 ,154.39,  78.01, 182. ,  136.3 , 244.18, 197.85],
                                    [162.89 ,119.27 ,478.6  ,337.96,  57.43, 106.89, 369.07, 549.61, 537.51, 385.5,
                                    297.06 ,146.6  ,364.05 ,  0.   ,357.24 , 45.18, 234.05, 307.72, 452.59 ,379.86,
                                    474.91, 560.82 ,192.84 ,299.27, 445.48, 385.41 ,473.86, 284.69 ,438.98 ,465.55],
                                    [429.79 ,295.37 ,340.33 , 94.87, 408.42 ,341.84, 295.21, 232.02, 238.13 ,120.62,
                                    307.56 ,221.96 ,174.16 ,357.24 ,  0.   ,312.61, 130.97, 231.7 , 215.26,  40.11,
                                    143.61 ,241.45 ,202.81 , 64.14, 108.91, 251.86, 129.36, 268.13, 414.4 , 112.2 ],
                                    [185.61 ,103.09 ,453.2  ,297.14 ,100.96, 109.66, 346. ,  509.07, 498.38, 341.08,
                                    279.81 ,107.79 ,328.69 , 45.18, 312.61 ,  0. ,  191.34, 270.28, 416.92, 336.09,
                                    432.67 ,520.22 ,150.48 ,254.19, 402.62, 356.59, 430.64, 262.15, 425.67, 421.54],
                                    [299.7  ,164.56 ,321.73 ,108.75, 281.63, 210.94, 238.61, 319.51, 311.85 ,206.42,
                                    216.81 , 91.92 ,162.11 ,234.05, 130.97 ,191.34,   0. ,  233.4 , 243.39 ,147.05,
                                    241.35 ,330.51 ,134.16 , 90.55, 211.44 ,219.15, 239.89, 179.3 , 351.46, 233.08],
                                    [451.09 ,330.15 ,546.62 ,300.73, 364.27, 363.11, 471.56, 459.68, 469.11 ,151.4,
                                    446.2  ,260.23 ,377.77, 307.72, 231.7,  270.28, 233.4  ,  0. ,  440.2  ,271.81,
                                    371.26 ,468.16 ,128.86, 183.24, 337.07, 446.47, 350.75 ,410.6 , 584.72 ,324.72],
                                    [446.48 ,345.3  ,143.59, 142.43, 486.43, 385.03 ,165.66, 135.82, 105.7  ,332.92,
                                    233.06 ,309.38 , 88.55 ,452.59, 215.26, 416.92 ,243.39, 440.2 ,   0. ,  178.13,
                                    133.33 ,145.6  ,373.13 ,257.17 ,145.24, 108.16 ,162.01, 209.02 ,260.62 ,188.61],
                                    [438.72 ,307.32 ,308.38 , 76.3 , 428.6 , 353.9,  273.31, 193.55, 198.31 ,154.87,
                                    295.76 ,238.8  ,147.67 ,379.86 , 40.11, 336.09 ,147.05 ,271.81 ,178.13,   0.,
                                    105.95, 203.3  ,237.29 , 99.62 , 71.69, 225.5,   96.02, 257.13, 391.47 , 86.02],
                                    [512.06, 389.01 ,276.83 ,139.63 ,519.94, 434.44, 282.51 , 88.68 ,100.9  ,236.52,
                                    331.27 ,329.51 ,162.63, 474.91, 143.61, 432.67, 241.35, 371.26, 133.33, 105.95,
                                     0.    ,97.86 ,343.,   205.55 , 34.71 ,225.18,  30.41, 297.64, 389.39 , 61.03],
                                    [580.02 ,466.  , 268.36 ,223.08, 602.34, 509.56, 311.19 , 11.31,  43.05, 329.11,
                                    376.72 ,414.24, 215.78, 560.82, 241.45, 520.22, 330.51, 468.16, 145.6,  203.3,
                                    97.86 ,  0. ,  439.4 , 302.88, 132.57, 253.73, 118.02, 349.36, 399.15, 146.87],
                                    [324.42 ,201.3 , 455.29, 231.95 ,250.22, 235.05 ,366.55 ,429.17, 428.57 ,196.35,
                                    328.8  ,133.54, 295.53, 192.84, 202.81, 150.48 ,134.16 ,128.86 ,373.13 ,237.29,
                                    343.  , 439.4 ,   0. ,  138.71, 308.97 ,352.49 ,331.86 ,296.28, 472.87 ,314.4 ],
                                    [386.65, 250.6 , 366.9  ,117.92, 352.44 ,295.97 ,303.37, 293.03, 295.85, 117.,
                                    298.01 ,173., 196.98, 299.27,  64.14, 254.19,  90.55, 183.24, 257.17 , 99.62,
                                    205.55 ,302.88, 138.71,   0.  , 171.13, 269.82, 193.2 , 258.86, 421.46 ,176.14],
                                    [490.18 ,363.93, 287.63, 115.38 ,491.9,  409.91 ,279.69 ,123.32, 133.09 ,204.97,
                                    320.56 ,301.22, 154.39 ,445.48, 108.91, 402.62, 211.44, 337.07, 145.24 , 71.69,
                                    34.71 ,132.57, 308.97 ,171.13 ,  0.  , 224.36 , 30.46, 284.92 ,391.52,  44.94],
                                    [351.59 ,269.14, 102.84, 157.72, 410.02, 302.34 , 58.31, 243.82, 213.6,  371.38,
                                    128.  , 255.2  , 78.01 ,385.41, 251.86, 356.59, 219.15, 446.47 ,108.16 ,225.5,
                                    225.18 ,253.73 ,352.49 ,269.82, 224.36 ,  0.,   249.1,  111.61 ,167.87, 269.09],
                                    [520.61 ,394.   ,305.59 ,145.8 , 520.99 ,440.07 ,305.5  ,110.45 ,128.16, 211.48,
                                    349.38 ,330.34 ,182.  , 473.86, 129.36, 430.64, 239.89, 350.75, 162.01 , 96.02,
                                    30.41 ,118.02 ,331.86, 193.2  , 30.46 ,249.1   , 0.  , 314.24, 415.03  ,31.3 ],
                                    [240.26 ,165.6  ,194.59, 181.   ,303.06, 193.6 ,  84.38, 338.56, 313.02 ,374.33,
                                    39.56 ,174.24 ,136.3 , 284.69 ,268.13 ,262.15 ,179.3,  410.6  ,209.02 ,257.13,
                                    297.64 ,349.36 ,296.28, 258.86, 284.92 ,111.61, 314.24,   0.,   176.65, 325.44],
                                    [339.87 ,323.02 ,137.36 ,319.6 , 442.  , 334.86, 119.28 ,391.01 ,356.24 ,531.,
                                    147.83 ,349.01 ,244.18 ,438.98, 414.4,  425.67, 351.46, 584.72, 260.62 ,391.47,
                                    389.39 ,399.15 ,472.87 ,421.46 ,391.52 ,167.87 ,415.03, 176.65 ,  0.   ,436.41],
                                    [522.01 ,392.26 ,331.69, 149.4,  514.6,  438.73, 324. ,  140.07, 159.33 ,182.33,
                                    362.01 ,324.78 ,197.85, 465.55, 112.2 , 421.54 ,233.08 ,324.72 ,188.61  ,86.02,
                                    61.03 ,146.87 ,314.4 , 176.14,  44.94 ,269.09,  31.3 , 325.44, 436.41   ,0.  ],
  
                         ])

    data['demands'] =[0, 40, 26, 12, 43, 40, 36, 12, 37, 24, 39, 11, 27, 18, 25, 14, 37, 10, 29, 18, 14, 29, 34, 29, 14, 40, 16, 4, 33, 21]
    data['num_vehicles'] = 3
    data['vehicle_capacities'] = [150,150, 150]

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