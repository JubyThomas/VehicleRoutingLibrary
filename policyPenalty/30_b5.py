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
                                                 
                    [  0.,   497.81, 459.74, 377.24,  91.05, 525.84, 435.44, 357.76 ,311.72 ,499.06,
                    473.51, 465.09, 231.87, 248.77, 240.1,  528.16, 478.91, 372.77, 345.86, 217.75,
                    538.64, 347.97, 335.72, 393.72, 198.72, 365.55, 491.49, 350.32, 332.36 ,528.17],
                    [497.81,   0.  , 197.71, 304.14, 447.8 , 108.05, 147.71, 216.91, 202.97,  81.34,
                    479.09, 101.08, 268.7 , 312.62, 308.97, 301.57, 156.16, 363.04, 469.31, 282.84,
                    241.21, 410.03, 361.4 , 442.89, 299.1 , 236.55, 202.2 , 187.26, 413.2 , 216.42],
                    [459.74, 197.71,   0.  , 131.87, 382.47, 120.  ,  58.82, 333.14, 166.59, 123.49,
                    285.94,  96.75, 283.35, 374.93, 221.65, 114.06,  46.17, 187.88, 303.42, 293.07,
                    83.49, 242.33, 200.55, 262.76, 287.51, 358.93,  31.78, 121.41, 250.77 , 68.71],
                    [377.24, 304.14, 131.87,   0.   ,290.04, 248.74, 159.11, 369.12, 166.42, 244.17,
                    184.46, 209.09, 265.33, 368.96 ,145.12, 153.45, 175.17,  59.14, 172.19, 269.26,
                    183.25, 111.  ,  69.35, 138.76 ,249.72, 394.66, 155.19, 139.52, 118.9 , 183.67],
                    [ 91.05, 447.8 , 382.47, 290.04 ,  0.  , 460.39, 365.33, 345.28, 249.44, 436.69,
                    383.26, 400.58, 200.26, 254.79 ,160.9 , 442.56, 406.54, 282.29, 257.73, 187.95,
                    458.26, 257.03, 245.28, 303.32 ,159.35, 359.09, 413.86, 282.14, 241.51, 449.87],
                    [525.84, 108.05, 120.  , 248.74 ,460.39,   0.  ,  98.98, 306.32, 214.11,  34.01,
                    404.04,  60.8 , 314.03, 382.88 ,304.08, 204.  ,  74.17, 306.49, 420.92, 326.92,
                    138.  , 359.74, 315.3 , 382.58 ,333.26, 329.19, 110.6 , 178.28, 366.83, 113.99],
                    [435.44, 147.71,  58.82, 159.11 ,365.33,  98.98,   0.  , 275.77, 127.39,  86.31,
                    331.41,  50.  , 239.85, 325.02 ,206.71, 172.8 ,  45.61, 218.23, 329.11, 251.03,
                    133.36, 268.42, 221.65, 297.32 ,250.71, 301.39,  79.81,  85.23, 273.76, 112.36],
                    [357.76, 216.91, 333.14, 369.12 ,345.28, 306.32, 275.77,   0.  , 203.04, 272.41,
                    550.6 , 261.82, 149.63, 117.64 ,279.16, 446.36, 313.6 , 415.23, 485.88 ,159.53,
                    407.64, 440.44, 394.08, 487.89 ,191.39,  26.08, 355.24, 233.65, 435.27 ,384.76],
                    [311.72, 202.97, 166.59, 166.42 ,249.44, 214.11, 127.39, 203.04,   0.  , 188.09,
                    347.75, 153.39, 117.  , 213.43 ,107.08, 266.99, 172.75, 212.8 , 294.42, 126.48,
                    250.06, 242.92, 194.59, 287.78 ,123.37, 228.4 , 196.86 , 45.18, 240.3 , 233.28],
                    [499.06,  81.34, 123.49, 244.17 ,436.69,  34.01,  86.31 ,272.41, 188.09,   0.,
                    409.25,  38.08, 283.18, 349.52 ,283.48, 221.23,  78.55 ,303.04 ,415.13, 296.34,
                    159.86, 354.27, 307.88, 381.22 ,304.35, 295.43, 122.8  ,156.16 ,359.95, 135.07],
                    [473.51, 479.09, 285.94, 184.46 ,383.26, 404.04, 331.41 ,550.6  ,347.75, 409.25,
                        0.  , 379.12, 432.48, 535.62 ,293.17, 223.43, 331.84 ,135.76 ,133.73, 433.04,
                    290.08, 129.45, 162.86,  80.05 ,407.16, 575.68, 294.43 ,323.95 ,143.29, 305.06],
                    [465.09, 101.08,  96.75, 209.09 ,400.58,  60.8 ,  50.   ,261.82 ,153.39,  38.08,
                    379.12,   0.  , 254.62, 328.24 ,246.03, 204.36,  58.14, 268.19 ,378.96, 267.24,
                    150.95, 318.35, 271.35, 347.13 ,272.68, 286.25, 104.74, 119.02 ,323.49, 126.59],
                    [231.87, 268.7 , 283.35, 265.33 ,200.26, 314.03, 239.85, 149.63 ,117.  , 283.18,
                    432.48, 254.62,   0.  , 103.94 ,142.27, 382.77, 285.38, 298.2  ,348.8 ,  14.21,
                    366.72, 311.13, 269.9 , 361.49 , 42.01, 167.87 ,313.17, 162.05 ,302.97, 349.18],
                    [248.77, 312.62, 374.93, 368.96 ,254.79, 382.88, 325.02, 117.64 ,213.43, 349.52,
                    535.62, 328.24, 103.94,   0.   ,243.4 , 480.11 ,369.09, 401.88 ,446.13, 102.62,
                    456.92, 412.41, 372.8 , 463.26 ,130.83, 119.23, 402.78, 256.53 ,403.25, 437.19],
                    [240.1 , 308.97, 221.65, 145.12 ,160.9 , 304.08, 206.71, 279.16 ,107.08, 283.48,
                    293.17, 246.03, 142.27, 243.4  ,  0.  , 290.06, 246.4 , 162.56 ,207.77, 140.93,
                    298.8 , 169.12, 130.51, 219.86 ,113.99, 301.48 ,253.14, 127.42 ,160.7 , 289.37],
                    [528.16, 301.57, 114.06, 153.45 ,442.56, 204. ,  172.8 , 446.36 ,266.99, 221.23,
                    223.43, 204.36, 382.77, 480.11 ,290.06,   0. ,  146.41, 183.31 ,290.41, 390.57,
                    70.41, 236.38, 213.21, 233.44 ,379.02, 472.27 , 99.76, 223.66 ,250.32,  91.18],
                    [478.91, 156.16,  46.17 ,175.17 ,406.54,  74.17 , 45.61, 313.6  ,172.75,  78.55,
                    331.84,  58.14, 285.38 ,369.09 ,246.4 , 146.41 ,  0.  , 232.5  ,347.33, 296.62,
                    94.54, 286.15, 242.62 ,308.42 ,296.12, 338.72 , 46.67, 129.71 ,293.74,  71.17],
                    [372.77, 363.04, 187.88 , 59.14 ,282.29, 306.49, 218.23, 415.23 ,212.8 , 303.04,
                    135.76, 268.19, 298.2  ,401.88 ,162.56, 183.31 ,232.5 ,   0.   ,116.06, 299.66,
                    227.79,  55.9 ,  37.05 , 80.05 ,275.54, 440.16, 208.08, 192.78 , 67.8 , 232.56],
                    [345.86, 469.31, 303.42 ,172.19 ,257.73, 420.92, 329.11, 485.88 ,294.42, 415.13,
                    133.73, 378.96, 348.8  ,446.13 ,207.77, 290.41, 347.33 ,116.06 ,  0.  , 345.62,
                    341.72,  61.19, 108.   , 63.63 ,315.43, 508.79, 324.14, 287.7  , 56.14, 347.97],
                    [217.75, 282.84, 293.07 ,269.26 ,187.95, 326.92, 251.03, 159.53 ,126.48, 296.34,
                    433.04, 267.24,  14.21 ,102.62 ,140.93, 390.57, 296.62, 299.66 ,345.62,   0.,
                    376.54, 310.04, 270.2 , 360.77, 32.57, 176.5 , 323.24 ,171.66 ,301.23 ,359.49],
                    [538.64, 241.21,  83.49, 183.25 ,458.26, 138. ,  133.36, 407.64, 250.06, 159.86,
                    290.08, 150.95, 366.72, 456.92 ,298.8 ,  70.41,  94.54, 227.79, 341.72, 376.54,
                        0.  , 283.49, 250.98, 289.91 ,370.74, 432.92 , 54.15, 204.88 ,295.39 , 24.84],
                    [347.97, 410.03, 242.33, 111.  , 257.03, 359.74 ,268.42, 440.44, 242.92, 354.27,
                    129.45, 318.35, 311.13, 412.41, 169.12, 236.38 ,286.15,  55.9 ,  61.19, 310.04,
                    283.49,   0.  ,  49.48,  51.31, 282.21, 464.35 ,263.58, 231.39,  16.49 ,288.46],
                    [335.72, 361.4 , 200.55,  69.35, 245.28, 315.3 , 221.65, 394.08, 194.59, 307.88,
                    162.86, 271.35 ,269.9 , 372.8 , 130.51, 213.21, 242.62,  37.05, 108. ,  270.2,
                    250.98,  49.48 ,  0.  ,  93.81, 244.44, 418.4 , 224.54, 181.91,  52.15, 252.67],
                    [393.72, 442.89, 262.76, 138.76, 303.32, 382.58 ,297.32, 487.89, 287.78, 381.22,
                    80.05, 347.13, 361.49, 463.26, 219.86, 233.44, 308.42,  80.05,  63.63, 360.77,
                    289.91,  51.31,  93.81 ,  0.  , 333.29, 512.2 , 279.33, 271.49,  63.69, 299.04],
                    [198.72, 299.1 , 287.51 ,249.72, 159.35, 333.26, 250.71, 191.39, 123.37, 304.35,
                    407.16, 272.68,  42.01 ,130.83, 113.99, 379.02, 296.12, 275.54, 315.43,  32.57,
                    370.74, 282.21, 244.44 ,333.29,   0.  , 208.9 , 318.56, 167.44, 272.64 ,355.35],
                    [365.55, 236.55, 358.93 ,394.66, 359.09, 329.19, 301.39,  26.08, 228.4 , 295.43,
                    575.68, 286.25, 167.87 ,119.23, 301.48, 472.27, 338.72, 440.16, 508.79, 176.5,
                    432.92, 464.35, 418.4  ,512.2 , 208.9 ,   0.  , 380.74, 259.62, 458.8 , 409.89],
                    [491.49, 202.2 ,  31.78, 155.19, 413.86, 110.6 ,  79.81, 355.24, 196.86, 122.8,
                    294.43, 104.74 ,313.17, 402.78, 253.14,  99.76 , 46.67, 208.08, 324.14, 323.24,
                    54.15, 263.58 ,224.54, 279.33, 318.56, 380.74 ,  0.  , 151.75, 273.25 , 37.  ],
                    [350.32, 187.26, 121.41 ,139.52, 282.14, 178.28 , 85.23, 233.65,  45.18, 156.16,
                    323.95, 119.02, 162.05 ,256.53, 127.42, 223.66 ,129.71, 192.78, 287.7 , 171.66,
                    204.88, 231.39, 181.91 ,271.49, 167.44, 259.62 ,151.75,   0.  , 231.84 ,188.28],
                    [332.36, 413.2 , 250.77 ,118.9 , 241.51, 366.83, 273.76, 435.27, 240.3 , 359.95,
                    143.29, 323.49, 302.97 ,403.25, 160.7 , 250.32, 293.74,  67.8 ,  56.14, 301.23,
                    295.39,  16.49,  52.15 , 63.69, 272.64, 458.8 , 273.25, 231.84,   0.  , 299.36],
                    [528.17, 216.42,  68.71 ,183.67, 449.87, 113.99 ,112.36, 384.76, 233.28 ,135.07,
                    305.06, 126.59, 349.18 ,437.19, 289.37,  91.18 , 71.17, 232.56, 347.97 ,359.49,
                    24.84, 288.46, 252.67 ,299.04, 355.35, 409.89 , 37.  , 188.28, 299.36 ,  0.  ]
                                 
                                ])
    
    data['demands'] =[0, 21, 28, 38, 40, 18, 36, 31, 30, 28, 21, 29, 31, 39, 30, 18,
                      29, 34, 27, 21, 38, 36, 27, 28, 36, 40, 21, 14, 34, 33] 
    

 
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