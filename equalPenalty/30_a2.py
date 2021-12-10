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
                                 [  0.  , 402.66 ,211.7,  139.9 , 126.78, 212.2 , 163.31, 187.86, 192.35, 497.04,
                                  529.27,  21.1,  380.55,  98.41, 434.56, 332.12,  88.28, 166.86, 443.6,  219.32,
                                   138. ,  134.24 ,396.92, 167.26, 143.34, 254.61, 456.45,  70.49, 416.08, 330.34],
                                [402.66,   0.,   236.18 ,454.47, 278.47, 311.82, 241.27, 263.43, 212.12, 280.11,
                                308.2 , 423.74, 139.28, 326.11, 187.18 ,123.69, 421.88, 255.55,  81.01, 297.48,
                                346.92, 409.34,   7.21, 349.5,  263.27, 384.16, 108.5,  342.29, 208.59, 207.87],
                                [211.7 , 236.18 ,  0.  , 221.94,  99.09, 265.34,  82.64, 200.04, 117.04, 438.75,
                                472.36, 230.26, 172.68, 186.64, 350.66, 131.75, 271.18, 163.49, 251.05 , 71.12,
                                113.56 ,288.21, 232.82, 263.41, 133.42, 344.37, 324.75, 181.84, 349.27, 284.48],
                                [139.9 , 454.47 ,221.94,   0. ,  193.85, 342.58, 225.79 ,303.29, 271.28, 606.93,
                                640.36, 136.01, 387.94 ,222.27 ,532.56, 353.04, 225.14 ,271.95, 472.65, 183.39,
                                108.42, 273.76, 450.32, 303.6,  237.01, 393.13, 531.61, 193.69, 520.48, 438.85],
                                [126.78, 278.47,  99.09, 193.85,   0. ,  184.,    37.22, 125.93,  77.67 ,415.11,
                                448.77, 147.49, 257.01,  88.2 , 338.75, 206.6,  173.44,  88.6 , 316.83, 140.19,
                                106.51, 189.28, 273.1,  170.5 ,  50.7,  257.29 ,342.48,  83.41, 327.56 ,248.21],
                                [212.2 , 311.82, 265.34, 342.58, 184. ,    0. ,  183.56,  66.31, 153.52 ,298.29,
                                328.2 , 226.91, 367.6,  120.42, 256.98, 312.82, 164.48, 101.87, 382.26, 321.7,
                                284.58, 124.33 ,304.65,  55.15, 135.53,  81.63 ,315.23, 148.92, 229.68 ,146.87],
                                [163.31, 241.27,  82.64, 225.79,  37.22, 183.56 ,  0. ,  119.42,  48.26 ,389.01,
                                422.83, 184.17, 224.95, 110.11, 309.01 ,172.89, 202.71 , 81.88, 280.59, 140.03,
                                129.47, 211.57, 235.92, 181.25,  51. ,  261.89, 307.25, 113.22 ,300.32, 224.76],
                                [187.86, 263.43, 200.04, 303.29, 125.93 , 66.31, 119.42 ,  0. ,   87.24, 309.38,
                                341.88, 206.39 ,304.83,  89.59, 248.97, 249.73, 169.43,  38.05, 328.01, 259.09,
                                231.17, 147.14, 256.46,  86.77 , 75.31 ,147.6 , 286.45, 117.61, 228.62, 142.58],
                                [192.35, 212.12, 117.04, 271.28,  77.67 ,153.52 , 48.26,  87.24,   0.,   340.8,
                                374.61, 213.41, 223.96 ,115.95, 261.77 ,169.01, 212.78,  56.59, 262.84 ,182.48,
                                177.61, 209.62 ,206.02, 164.79,  51.24 ,234.79 ,266.51, 130.19, 252.24, 176.71],
                                [497.04 ,280.11, 438.75, 606.93, 415.11, 298.29, 389.01, 309.38, 340.8 ,   0.,
                                33.84, 515.02, 415.09, 398.63 , 94.89 ,382.46, 462.53, 335.97 ,354.13, 509.87,
                                518.4,  421.38, 276.01, 353.33, 369.93, 315.36, 183.6  ,426.98 , 89.55 ,168.32],
                                [529.27, 308.2 , 472.36, 640.36, 448.77 ,328.2,  422.83 ,341.88 ,374.61,  33.84,
                                0. ,  547.03 ,444.71, 430.89, 126.13 ,413.69, 492.67, 369.11 ,380.03, 543.48,
                               552.2  ,450.37, 304.47, 383.01, 403.35 ,340.39, 207.63, 459.4,  123.31 ,201.56],
                                [ 21.1,  423.74 ,230.26 ,136.01 ,147.49, 226.91 ,184.17, 206.39, 213.41, 515.02,
                                547.03 ,  0. ,  400.13, 116.81, 454.01, 352.23,  89.2 , 186.85 ,464.28 ,233.81,
                                149.97, 139.06, 418.01, 179.47, 164.22 ,264.2,  477.21,  89.9,  434.94, 348.97],
                                [380.55, 139.28, 172.68, 387.94 ,257.01, 367.6,  224.95 ,304.83 ,223.96 ,415.09,
                                444.71 ,400.13 ,  0.  , 333.22 ,320.45 , 55.17 ,427.6  ,280.17,  99.49, 207.69,
                                282.  , 432.16 ,141.66 ,388.  , 268.22, 448.38, 247.63 ,338.11 ,336.29 ,310.63],
                                [ 98.41, 326.11 ,186.64 ,222.27,  88.2 , 120.42, 110.11 , 89.59 ,115.95 ,398.63,
                                430.89 ,116.81 ,333.22  , 0.  , 337.65, 279.65 , 96.84,  73.82 ,378.73, 225.44,
                                171.05 ,101.86, 319.71 , 89.02,  65.92, 180.14, 366.81,  29.21 ,318.13 ,232.17],
                                [434.56 ,187.18 ,350.66, 532.56, 338.75, 256.98 ,309.01, 248.97, 261.77,  94.89,
                                126.13, 454.01, 320.45 ,337.65  , 0.  , 287.7 , 414.37 ,268.22, 263.91, 421.57,
                                437.95, 380.47, 182.62, 310.85 ,298.06 ,297.62, 102.08, 364.18 , 34.89, 110.42],
                                [332.12, 123.69, 131.75, 353.04 ,206.6 , 312.82, 172.89, 249.73, 169.01 ,382.46,
                                413.69 ,352.23 , 55.17, 279.65 ,287.7  ,  0.  , 374.86, 225.09, 119.64, 181.18,
                                244.97, 377.76 ,123.22, 332.85 ,214.22 ,393.81, 228.97, 286. ,  299.13, 263.97],
                                [ 88.28, 421.88, 271.18, 225.14, 173.44, 164.48, 202.71, 169.43, 212.78, 462.53,
                                492.67 , 89.2 , 427.6  , 96.84 ,414.37, 374.86 ,  0.  , 166.77 ,475.56 ,294.61,
                                220.48 , 52.33, 415.36 ,110.11 ,162.54 ,182.78, 455.59,  90.14, 390.22, 304.54],
                                [166.86, 255.55, 163.49 ,271.95 , 88.6 , 101.87 , 81.88,  38.05,  56.59, 335.97,
                                369.11 ,186.85, 280.17 , 73.82 ,268.22, 225.09, 166.77,   0. ,  313.47, 221.18,
                                194.64 ,156.26, 248.92, 108.21 , 37.95, 181.43, 293.02,  97.14, 251.63, 167.66],
                                [443.6 ,  81.01, 251.05, 472.65, 316.83, 382.26 ,280.59, 328.01, 262.84, 354.13,
                                380.03 ,464.28,  99.49, 378.73, 263.91, 119.64 ,475.56, 313.47 ,  0.,   298.94,
                                364.49, 469.69 , 87.46, 414.77, 313.2 , 457.91 ,173.01, 390.37, 287.77, 288.06],
                                [219.32, 297.48,  71.12, 183.39, 140.19, 321.7  ,140.03, 259.09, 182.48, 509.87,
                                543.48, 233.81, 207.69, 225.44, 421.57, 181.18 ,294.61, 221.18, 298.94 ,  0.,
                                87.21, 322.12, 295.  , 310.58, 186.19, 397.15 ,391.75, 212.06, 420.39, 354.49],
                                [138.,   346.92, 113.56, 108.42, 106.51, 284.58, 129.47 ,231.17, 177.61 ,518.4,
                                552.2 , 149.97, 282. ,  171.05, 437.95, 244.97, 220.48, 194.64, 364.49 , 87.21,
                                0. ,  255.93 ,342.99, 260.07, 156.73, 351.01, 427.95, 149.63, 429.79, 353.11],
                               [134.24, 409.34, 288.21, 273.76, 189.28, 124.33, 211.57, 147.14, 209.62, 421.38,
                                450.37, 139.06, 432.16, 101.86, 380.47, 377.76,  52.33, 156.26, 469.69, 322.12,
                                255.93 ,  0. ,  402.48,  69.63, 164.2,  131.02, 431.09, 110.06, 353.9 , 270.05],
                                [396.92 ,  7.21, 232.82, 450.32, 273.1 , 304.65, 235.92, 256.46, 206.02, 276.01,
                                304.47, 418.01, 141.66, 319.71, 182.62, 123.22, 415.36, 248.92,  87.46, 295.,
                                342.99, 402.48,   0. ,  342.47, 257.12, 376.95, 107.02, 336.15, 203.27, 201.02],
                                [167.26, 349.5 , 263.41, 303.6,  170.5,   55.15, 181.25,  86.77, 164.79 ,353.33,
                                383.01, 179.47 ,388.  ,  89.02, 310.85, 332.85, 110.11, 108.21, 414.77, 310.58,
                                260.07 , 69.63, 342.47 ,  0.  , 130.25 , 91.44, 363.79, 113.14, 284.34, 200.44],
                                [143.34, 263.27, 133.42 ,237.01,  50.7,  135.53,  51.,    75.31,  51.24 ,369.93,
                                403.35 ,164.22, 268.22  ,65.92, 298.06, 214.22,162.54 , 37.95, 313.2,  186.19,
                                    156.73 ,164.2 , 257.12 ,130.25 ,  0.   ,211.99, 313.21 , 79.03, 283.94 ,201.88],
                                    [254.61 ,384.16 ,344.37 ,393.13, 257.29 , 81.63, 261.89 ,147.6 , 234.79, 315.36,
                                    340.39 ,264.2 , 448.38 ,180.14 ,297.62, 393.81, 182.78 ,181.43, 457.91, 397.15,
                                    351.01 ,131.02 ,376.95  ,91.44, 211.99 ,  0.  , 372.21 ,204.53, 265.42, 195.6 ],
                                    [456.45 ,108.5  ,324.75, 531.61, 342.48, 315.23 ,307.25 ,286.45 ,266.51 ,183.6,
                                    207.63 ,477.21 ,247.63 ,366.81, 102.08, 228.97 ,455.59 ,293.02 ,173.01 ,391.75,
                                427.95, 431.09 ,107.02 ,363.79, 313.21, 372.21 ,  0.   ,389.08 ,133.87 ,176.85],
                                [ 70.49 ,342.29 ,181.84 ,193.69 , 83.41 ,148.92 ,113.22 ,117.61 ,130.19 ,426.98,
                                459.4  , 89.9  ,338.11 , 29.21 ,364.18 ,286.   , 90.14 , 97.14 ,390.37 ,212.06,
                                149.63 ,110.06 ,336.15 ,113.14 , 79.03 ,204.53 ,389.08 ,  0.   ,345.61 ,259.94],
                                [416.08 ,208.59 ,349.27 ,520.48 ,327.56 ,229.68 ,300.32 ,228.62 ,252.24 , 89.55,
                                123.31 ,434.94 ,336.29 ,318.13 , 34.89 ,299.13 ,390.22 ,251.63 ,287.77 ,420.39,
                                429.79 ,353.9  ,203.27 ,284.34 ,283.94 ,265.42 ,133.87 ,345.61  , 0.   , 86.31],
                                [330.34 ,207.87 ,284.48 ,438.85 ,248.21 ,146.87 ,224.76 ,142.58 ,176.71 ,168.32,
                                201.56 ,348.97 ,310.63 ,232.17 ,110.42 ,263.97 ,304.54 ,167.66 ,288.06 ,354.49,
                                353.11 ,270.05 ,201.02 ,200.44 ,201.88 ,195.6  ,176.85 ,259.94  ,86.31 ,  0.  ],
                                    ])
    
    data['demands'] =[0, 23, 27, 22, 13, 18, 40, 31, 38, 26, 21, 18, 36, 31, 40, 34, 26, 29, 31, 38, 18, 21, 39, 34, 15, 19, 23, 40, 21, 27]

    data['num_vehicles'] = 3
    data['vehicle_capacities'] = [150,150, 150]


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