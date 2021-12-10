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

    data = {}
    data['distance_matrix'] =np.array([
                                [  0. ,  261.34 ,344.54, 388.15 ,175.07, 454.36, 475.11, 175.51, 305.81, 387.66,
                                323.89, 133.06, 368.28, 495.  , 327.39, 448.56, 329.2,  228.97, 454.59, 433.54,
                                29.07, 385.75, 227.65, 251.92 ,404.75, 246.16, 286.9 ,  95.86, 173.48, 216.97],
                                [261.34,   0.,    90.09, 190.08 , 89.16, 273.91, 296.18, 104.66, 122.1,  140.23,
                                166.6 , 166.96, 285.28, 317.59 , 77.  , 194.28, 303.57, 372.25, 263.94, 216.12,
                                277.46, 125.87, 373.16,  84.76 ,209.75, 162.19,  77.1 , 221.05, 100.22 ,386.85],
                                [344.54,  90.09 ,  0. ,  128.84 ,177.88, 210.04, 230.87, 173.83, 171.41,  51.35,
                                141.42, 233.45, 267.11, 251.21 , 18.44, 105.  , 306.06, 420.19, 195.9,  138.42,
                                357.56,  76.01, 421.64, 112.06 ,146.58, 182.15,  74.69, 290.13, 190.25, 440.78],
                                [388.15, 190.08, 128.84,   0.   ,256.33,  84.06, 106.17, 217.08, 296.8  , 99.33,
                                71.56, 256.87, 163.42, 127.53 ,126.62, 137.32, 222.31, 388.31 , 74.33 , 51.22,
                                391.95, 194.65, 390.49, 141.17 , 19.7 , 154.77, 118. ,  308.33, 275.04 ,417.16],
                                [175.07,  89.16, 177.88 ,256.33 ,  0.  , 336.29, 358.71,  67.23, 147.87 ,226.13,
                                211.57, 111.  , 308.38, 380.18 ,162.81, 282.76, 305.41 ,321.4 , 330.11 ,291.69,
                                194. ,  211.01, 321.7 , 118.85 ,275.61, 168.44, 139.17, 153.84,  21.26 ,328.9 ],
                                [454.36, 273.91, 210.04,  84.06 ,336.29,   0.  ,  22.47, 290.63, 380.02 ,172.05,
                                131.14, 321.31, 152.5 ,  44.01 ,209.75, 188.32, 224.16, 416.69,  20.02 , 83.01,
                                454.44, 269.05, 419.23 ,218.28 , 64.64, 209.75, 200.24, 367.56, 355.76, 449.38],
                                [475.11, 296.18, 230.87, 106.17 ,358.71,  22.47,   0.  , 312.55, 401.37, 191.17,
                                152.64, 342.05, 161.44 , 21.54 ,231.21, 202.61, 234.07 ,430.53,  35.61,  99.64,
                                474.61, 287.8 , 433.13, 240.6  , 86.61, 229.79, 222.7 , 387.39, 378.21, 463.88],
                                [175.51, 104.66, 173.83, 217.08 , 67.23, 290.63, 312.55,   0.  , 203.06, 213.41,
                                160.43,  62.43, 244.43, 333.57 ,155.8 , 275.43, 238.25, 269.7 , 287.63 ,259.79,
                                185.18, 228.  , 270.43 , 77.1  ,235.01, 103.71, 111.43, 117.8 ,  87.66 ,282.66],
                                [305.81, 122.1 , 171.41, 296.8  ,147.87, 380.02, 401.37, 203.06,   0.  , 218.3,
                                286.25, 257.53, 407.19, 422.04 ,170.29, 248.6 , 424.38, 468.73, 366.81, 309.63,
                                329.85, 142.41, 469.12, 206.55 ,315.63, 282.11, 196.09, 301.48, 138.51 ,476.77],
                                [387.66, 140.23,  51.35,  99.33 ,226.13, 172.05, 191.17, 213.41, 218.3 ,   0.,
                                137.54, 269.36, 256. ,  210.28 , 63.51,  63.13, 305.37, 441.9 , 155.57,  93.43,
                                398.59,  97.1 , 443.61, 142.44 ,113.22, 198.88, 103.81, 325.85, 239.76 ,465.56],
                                [323.89, 166.6 , 141.42 , 71.56 ,211.57, 131.14, 152.64, 160.43, 286.25 ,137.54,
                                    0.   ,191.05, 126.13 ,173.4  ,130.  , 192.01, 168.02, 318.25, 130.75 ,122.64,
                                325.35, 217.02, 320.34 , 93.15 , 83.93,  83.77,  90.21, 239.87, 232.11 ,346.3 ],
                                [133.06, 166.96, 233.45, 256.87 ,111.  , 321.31, 342.05,  62.43, 257.53 ,269.36,
                                191.05 ,  0.  , 244.31, 362.01 ,215.13, 332.24, 219.57, 211.29, 321.8  ,303.94,
                                135.09 ,290.27, 211.75, 127.42 ,272.81, 113.75, 165.58,  56.73, 126.15 ,221.69],
                                [368.28, 285.28, 267.11 ,163.42 ,308.38, 152.5 , 161.44, 244.43, 407.19 ,256.,
                                126.13, 244.31 ,  0.  , 172.05 ,256.12, 300.7 ,  72.8 , 274.7 , 167.63, 204.63,
                                359.68, 342.13, 277.46, 202.47 ,159.77, 140.72, 212.72, 272.86, 329.62 ,309.81],
                                [495.  , 317.59, 251.21 ,127.53 ,380.18,  44.01 , 21.54, 333.57, 422.04 ,210.28,
                                173.4,  362.01, 172.05  , 0.   ,252.03, 217.85, 244.83, 444.18,  55.32 ,117.61,
                                493.97, 306.42, 446.84 ,261.98 ,107.91, 249.24, 244.23, 406.51, 399.71 ,478.1 ],
                                [327.39,  77.  ,  18.44 ,126.62 ,162.81, 209.75, 231.21, 155.8 , 170.29 , 63.51,
                                130.   ,215.13, 256.12 ,252.03 ,  0.  , 121.33, 292.06, 402.18, 196.95 ,143.,
                                339.94 , 90.67 ,403.6   ,93.81 ,145.34, 165.25,  57.01, 271.83, 176.26 ,422.52],
                                [448.56, 194.28, 105.   ,137.32 ,282.76, 188.32, 202.61, 275.43, 248.6 ,  63.13,
                                192.01, 332.24, 300.7  ,217.85 ,121.33,   0.  , 356.8 , 503.21, 168.88, 106.1,
                                460.42, 108.08, 505.   ,205.53 ,144.  , 260.  , 166.82, 388.8 , 294.44, 527.64],
                                [329.2  ,303.57, 306.06 ,222.31 ,305.41, 224.16, 234.07, 238.25, 424.38 ,305.37,
                                168.02, 219.57,  72.8  ,244.83 ,292.06, 356.8 ,   0.  , 204.92, 238.19 ,268.41,
                                315.9 , 382.03, 207.78 ,219.02 ,222.77, 142.44, 240.52, 233.94, 325.89 ,240.93],
                                [228.97, 372.25, 420.19 ,388.31 ,321.4 , 416.69, 430.53, 269.7 , 468.73 ,441.9,
                                318.25, 211.29, 274.7  ,444.18 ,402.18, 503.21, 204.92,   0.  , 426.91 ,439.36,
                                202.43, 487.88,   3.   ,308.59 ,396.23, 243.21, 345.58, 172.42 ,334.29 , 37.05],
                                [454.59, 263.94, 195.9  , 74.33 ,330.11,  20.02,  35.61, 287.63 ,366.81 ,155.57,
                                130.75 ,321.8 , 167.63 , 55.32 ,196.95, 168.88, 238.19, 426.91 ,  0.   , 64.41,
                                455.95 ,252.2 , 429.38 ,213.41 , 54.71, 212.27, 192.28, 369.88, 349.07 ,458.88],
                                [433.54, 216.12, 138.42 , 51.22 ,291.69,  83.01,  99.64, 259.79, 309.63 , 93.43,
                                122.64, 303.94, 204.63 ,117.61 ,143.  , 106.1 , 268.41, 439.36 , 64.41  , 0.,
                                438.88, 188.85, 441.56 ,182.72 , 45.65, 205.29, 153.01, 356.75, 308.88 ,468.37],
                                [ 29.07, 277.46, 357.56 ,391.95 ,194.  , 454.44, 474.61, 185.18, 329.85 ,398.59,
                                325.35 ,135.09, 359.68 ,493.97 ,339.94, 460.42 ,315.9 , 202.43 ,455.95 ,438.88,
                                    0.  , 402.94, 200.96 ,259.62 ,407.75, 244.82 ,296.14,  88.09 ,194.83 ,188.82],
                                [385.75, 125.87,  76.01 ,194.65 ,211.01, 269.05, 287.8 , 228. ,  142.41 , 97.1,
                                217.02, 290.27, 342.13 ,306.42 , 90.67, 108.08, 382.03, 487.88 ,252.2  ,188.85,
                                402.94,   0. ,  489.12 ,180.28 ,209.82, 255.22, 146.54, 345.65, 216.6  ,505.94],
                                [227.65, 373.16, 421.64 ,390.49 ,321.7 , 419.23, 433.13, 270.43, 469.12 ,443.61,
                                320.34, 211.75, 277.46 ,446.84, 403.6  ,505.  , 207.78 ,  3.  , 429.38 ,441.56,
                                200.96, 489.12,   0.   ,309.96 ,398.5 , 245. ,  347.05, 172.24, 334.43 , 34.06],
                                [251.92,  84.76, 112.06 ,141.17, 118.85, 218.28, 240.6 ,  77.1 , 206.55, 142.44,
                                93.15 ,127.42, 202.47 ,261.98 , 93.81, 205.53, 219.02, 308.59 ,213.41, 182.72,
                                259.62, 180.28 ,309.96 ,  0.   ,159.71,  79.1 ,  38.9 , 183.61, 139.13, 328.72],
                                [404.75, 209.75 ,146.58 , 19.7  ,275.61 , 64.64 , 86.61 ,235.01, 315.63 ,113.22,
                                83.93, 272.81 ,159.77 ,107.91 ,145.34 ,144.  , 222.77, 396.23 , 54.71 , 45.65,
                                407.75 ,209.82 ,398.5  ,159.71  , 0.  , 167.69, 137.57, 323.21, 294.45, 426.08],
                                [246.16 ,162.19 ,182.15 ,154.77 ,168.44, 209.75, 229.79, 103.71, 282.11, 198.88,
                                83.77, 113.75 ,140.72, 249.24 ,165.25, 260.  , 142.44, 243.21, 212.27, 205.29,
                                244.82, 255.22 ,245.  ,  79.1  ,167.69 ,  0.  , 108.68, 157.84, 189.59, 268.26],
                                [286.9 ,  77.1  , 74.69, 118.   ,139.17, 200.24, 222.7,  111.43, 196.09, 103.81,
                                90.21, 165.58, 212.72 ,244.23 , 57.01, 166.82, 240.52, 345.58, 192.28, 153.01,
                                296.14, 146.54, 347.05 , 38.9  ,137.57, 108.68 ,  0.  , 222.04, 157.29 ,366.66],
                                [ 95.86, 221.05 ,290.13 ,308.33 ,153.84, 367.56 ,387.39, 117.8,  301.48, 325.85,
                                239.87,  56.73 ,272.86 ,406.51 ,271.83, 388.8 , 233.94, 172.42, 369.88, 356.75,
                                88.09, 345.65 ,172.24 ,183.61 ,323.21, 157.84, 222.04,   0. ,  164.07 ,175.79],
                                [173.48 ,100.22 ,190.25 ,275.04 , 21.26, 355.76, 378.21,  87.66, 138.51, 239.76,
                                232.11, 126.15 ,329.62, 399.71 ,176.26, 294.44, 325.89, 334.29, 349.07, 308.88,
                                194.83, 216.6 , 334.43, 139.13 ,294.45, 189.59, 157.29, 164.07,   0.,   339.85],
                                [216.97, 386.85, 440.78, 417.16 ,328.9 , 449.38, 463.88 ,282.66, 476.77 ,465.56,
                                346.3,  221.69 ,309.81, 478.1  ,422.52, 527.64, 240.93 , 37.05, 458.88, 468.37,
                                188.82, 505.94,  34.06, 328.72 ,426.08, 268.26, 366.66 ,175.79, 339.85  , 0.  ],
     
                                ])
    
    data['demands'] =[0, 21, 17, 12, 33, 37, 21, 18, 40, 36, 29, 31, 39, 31, 36, 21, 17, 14, 17, 26, 34, 22, 19, 21, 29, 24, 36, 40, 20, 12]
 
   
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