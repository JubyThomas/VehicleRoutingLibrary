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
                                [  0. ,  658.62, 540.68, 398.08, 411.65, 298.77, 320.05, 293.17, 588.54, 541.5,
                                464.57 ,399.62, 472.13, 544.21, 555.13, 464.39, 643.05, 682.02 ,475.8 , 456.42,
                                351.4 , 694.11, 521.49, 727.39, 358.23, 477.46, 648.47 ,619.64 ,521.14 ,584.59],
                                [658.62,   0.  , 177.05, 268.29, 251.2 , 434.28, 347.47 ,420.2  ,158.67 ,183.83,
                                196.46, 345.12, 300.62, 289.75, 128.69, 375.22, 210.01  ,78.77 ,348.9  ,202.7,
                                382.97, 116.84, 330.94,  74.95, 328.79, 231.02, 136.37 ,193.65 ,145.62 , 74.33],
                                [540.68, 177.05,   0. ,  149.49, 194.56, 276.25, 273.75 ,368.02 ,256.7  ,246.48,
                                112.81, 336.66, 123.76, 356.76,  54.08, 402.49, 333.47 ,156.17 ,381.37 ,128.55,
                                354.38, 280.94, 385.25, 217.68, 184.01,  65.62 ,108.07 ,307.77 , 76.69 ,126.29],
                                [398.08, 268.29, 149.49,   0.  ,  87.21, 177.55 ,130.86 ,227.97 ,255.67 ,220.55,
                                72.92, 225.58, 149.6 , 301.1 , 157.05, 307.74 ,332.56 ,284.02 ,293.49 , 72.01,
                                227.66, 332.7 , 314.  , 332.25,  69.53,  97.84 ,253.95 ,304.28 ,124.62 ,194.49],
                                [411.65, 251.2 , 194.56,  87.21,   0.  , 248.81 , 96.33 ,178.9  ,190.   ,147.8,
                                82.07, 145.8 , 232.21, 215.24, 176.3 , 223.01 ,260.97 ,290.41 ,207.28 , 66.71,
                                159.83, 285.41, 226.8 , 323.79, 149. ,  165.6  ,281.77 ,233.35 ,134.83 ,179.61],
                                [298.77, 434.28, 276.25, 177.55, 248.81,   0.   ,220.48 ,291.99 ,432.09 ,394.24,
                                247.86, 347.33, 180.01, 459.01, 309.39, 435.97 ,507.41 ,431.5  ,429.85 ,249.3,
                                324.89, 509.14, 461.46, 489.36, 108.71, 211.12 ,381.24 ,479.3  ,289.25 ,363.58],
                                [320.05, 347.47, 273.75, 130.86,  96.33, 220.48 ,  0.   , 97.19 ,269.7  ,223.22,
                                168.62, 127.09, 274.4 , 252.24, 266.1 , 215.64 ,330.35 ,385.15 ,211.03 ,155.6,
                                108.05, 374.07, 246.47, 420.1,  154.69, 228.42 ,370.52 ,304.93 ,225.82 ,275.86],
                                [293.17, 420.2 , 368.02, 227.97, 178.9 , 291.99  ,97.19 ,  0.   ,312.64 ,265.83,
                                258.85, 106.89, 370.55, 251.29, 355.01, 176.18 ,356.14 ,467.47 ,184.04 ,244.39,
                                58.25, 424.68, 228.95, 494.79, 246.88, 325.3  ,460.42 ,335.19 ,313.65 ,353.28],
                                [588.54, 158.67, 256.7 , 255.67, 190.  , 432.09, 269.7  ,312.64 ,  0.   , 47.3,
                                193.3 , 216.67, 358.65, 131.52, 205.  , 224.54 , 79.48 ,234.69 ,197.71 ,186.68,
                                263.  , 113.28, 173.88, 225.4,  325.15, 277.99 ,274.81 , 51.92 ,184.35 ,140.43],
                                [541.5 , 183.83, 246.48, 220.55, 147.8  ,394.24 ,223.22 ,265.83 , 47.3  ,  0.,
                                165.52, 172.29, 336.76 ,114.13, 199.15 ,191.4  ,113.36 ,254.07 ,165.08 ,156.02,
                                217.33, 159.03 ,150.66 ,255.78, 289.31 ,256.83 ,283.24 , 85.56 ,170.36 ,145.99],
                                [464.57, 196.46 ,112.81 , 72.92,  82.07 ,247.86 ,168.62 ,258.85 ,193.3  ,165.52,
                                    0.  , 224.38 ,172.74, 263.,    97.5  ,294.15 ,272.35 ,219.79 ,274.83 , 15.81,
                                241.83 ,261.28, 284.96, 262.92, 139.2   ,95.34 ,202.14 ,244.2  , 58.05 ,122.26],
                                [399.62 ,345.12, 336.66, 225.58, 145.8 , 347.33 ,127.09 ,106.89 ,216.67 ,172.29,
                                224.38 ,  0.  , 374.89 ,144.59, 309.05,  88.65 ,251.54 ,403.97 , 85.48 ,208.57,
                                49.65 ,329.95, 126.15 ,419.79, 271.31 ,311.36 ,412.01 ,232.36 ,268.26 ,287.38],
                                [472.13 ,300.62 ,123.76 ,149.6 , 232.21 ,180.01 ,274.4  ,370.55 ,358.65 ,336.76,
                                172.74 ,374.89 ,  0.   ,435.26, 173.83 ,455.19 ,437.98 ,274.76 ,439.14 ,185.13,
                                376.4  ,400.5 , 454.69, 339. ,  129.5  , 80.65 ,213.38 ,410.55 ,176.78 ,244.07],
                                [544.21 ,289.75, 356.76, 301.1,  215.24 ,459.01 ,252.24 ,251.29 ,131.52 ,114.13,
                                263.  , 144.59, 435.26 ,  0. ,  311.88 ,106.57 ,123.47 ,364.2  , 82.01 ,250.13,
                                193.3 , 229.43,  44.28 ,356.72, 364.02 ,358.11 ,397.16 ,115.32 ,280.07 ,259.85],
                                [555.13, 128.69,  54.08 ,157.05, 176.3  ,309.39 ,266.1  ,355.01 ,205.   ,199.15,
                                97.5  ,309.05 ,173.83, 311.88,   0.   ,366.99 ,280.61 ,127.19 ,344.22 ,110.92,
                                333.29 ,227.61 ,343.38, 180.25, 207.86, 102.42 ,105.48 ,255.44 , 41.62 , 72.24],
                                [464.39, 375.22, 402.49 ,307.74, 223.01, 435.97 ,215.64 ,176.18 ,224.54 ,191.4,
                                294.15 , 88.65, 455.19 ,106.57, 366.99 ,  0.   ,229.97 ,443.65  ,26.83 ,278.79,
                                121.26 ,331.49 , 66.48 ,446.84, 358.02 ,386.5  ,464.16 ,219.74 ,328.82 ,330.44],
                                [643.05 ,210.01, 333.47, 332.56, 260.97, 507.41 ,330.35 ,356.14 , 79.48 ,113.36,
                                272.35, 251.54, 437.98 ,123.47, 280.61, 229.97  , 0.   ,288.78 ,204.79 ,265.07,
                                300.99, 117.39, 166.4 , 262.99, 401.76, 357.34 ,339.35  ,28.3  ,262.91 ,212.02],
                                [682.02,  78.77, 156.17 ,284.02, 290.41, 431.5  ,385.15 ,467.47 ,234.69 ,254.07,
                                219.79, 403.97, 274.76, 364.2 , 127.19, 443.65 ,288.78 ,  0.   ,417.97 ,230.67,
                                436.95, 190.44 ,403.96,  66.6 , 333.73, 220.43  ,72.2  ,272.12 ,161.9  ,116.62],
                                [475.8 , 348.9 , 381.37 ,293.49, 207.28, 429.85 ,211.03 ,184.04 ,197.71 ,165.08,
                                274.83,  85.48, 439.14 , 82.01, 344.22,  26.83 ,204.79 ,417.97 ,  0.   ,259.76,
                                126.59, 304.87,  46.52 ,420.27, 347.13, 368.42 ,440.   ,193.61 ,306.8  ,305.45],
                                [456.42, 202.7,  128.55 , 72.01,  66.71 ,249.3  ,155.6  ,244.39 ,186.68 ,156.02,
                                15.81, 208.57, 185.13 ,250.13, 110.92, 278.79 ,265.07 ,230.67 ,259.76  , 0.,
                                226.29, 260.76, 270.9 , 271.15, 140.76, 109.33, 216.12 ,236.8  , 70.23 ,128.41],
                                [351.4 , 382.97, 354.38 ,227.66, 159.83, 324.89, 108.05  ,58.25 ,263.   ,217.33,
                                241.83 , 49.65, 376.4 , 193.3 , 333.29, 121.26 ,300.99 ,436.95 ,126.59 ,226.29,
                                    0.  , 376.11, 170.94, 457.92, 261.96, 320.62 ,438.21 ,281.3  ,291.73 ,320.69],
                                [694.11, 116.84, 280.94, 332.7 , 285.41, 509.14 ,374.07 ,424.68 ,113.28 ,159.03,
                                261.28, 329.95, 400.5  ,229.43, 227.61, 331.49 ,117.39 ,190.44 ,304.87 ,260.76,
                                376.11,   0. ,  273.71, 151.16, 400.46, 323.82 ,253.14 ,114.74 ,227.67 ,156.43],
                                [521.49, 330.94, 385.25 ,314.  , 226.8 , 461.46 ,246.47 ,228.95 ,173.88 ,150.66,
                                284.96, 126.15, 454.69 , 44.28, 343.38,  66.48 ,166.4  ,403.96 , 46.52 ,270.9,
                                170.94, 273.71,   0.   ,399.26 ,372.63, 380.17 ,433.21 ,159.54 ,308.95 ,296.28],
                                [727.39 , 74.95 ,217.68, 332.25 ,323.79, 489.36 ,420.1  ,494.79 ,225.4  ,255.78,
                                262.92, 419.79 ,339.  , 356.72 ,180.25 ,446.84 ,262.99 , 66.6  ,420.27 ,271.15,
                                457.92 ,151.16 ,399.26,   0.   ,387.77 ,279.65, 138.22 ,252.07 ,207.64 ,144.35],
                                [358.23 ,328.79 ,184.01,  69.53 ,149.   ,108.71, 154.69 ,246.88 ,325.15 ,289.31,
                                139.2  ,271.31 ,129.5 , 364.02 ,207.86 ,358.02, 401.76 ,333.73 ,347.13 ,140.76,
                                261.96 ,400.46 ,372.63 ,387.77  , 0.   ,119.51 ,292.06 ,373.5  ,183.17 ,256.7 ],
                                [477.46 ,231.02 , 65.62 , 97.84 ,165.6  ,211.12 ,228.42 ,325.3  ,277.99 ,256.83,
                                95.34 ,311.36 , 80.65 ,358.11 ,102.42 ,386.5  ,357.34 ,220.43 ,368.42 ,109.33,
                                320.62 ,323.82 ,380.17 ,279.65 ,119.51 ,  0.   ,173.07 ,329.89 , 97.25 ,168.01],
                                [648.47 ,136.37 ,108.07 ,253.95 ,281.77, 381.24 ,370.52 ,460.42 ,274.81 ,283.24,
                                202.14 ,412.01 ,213.38 ,397.16 ,105.48, 464.16 ,339.35 , 72.2  ,440.   ,216.12,
                                438.21 ,253.14 ,433.21 ,138.22 ,292.06, 173.07 ,  0.   ,318.94 ,147.03 ,137.31],
                                [619.64 ,193.65 ,307.77 ,304.28 ,233.35, 479.3  ,304.93 ,335.19 , 51.92 , 85.56,
                                244.2  ,232.36 ,410.55 ,115.32 ,255.44 ,219.74 , 28.3  ,272.12 ,193.61 ,236.8,
                                281.3  ,114.74 ,159.54 ,252.07 ,373.5  ,329.89 ,318.94  , 0.   ,236.14 ,188.38],
                                [521.14 ,145.62 , 76.69 ,124.62 ,134.83 ,289.25 ,225.82 ,313.65 ,184.35 ,170.36,
                                58.05 ,268.26, 176.78 ,280.07 , 41.62 ,328.82 ,262.91 ,161.9  ,306.8  , 70.23,
                                291.73 ,227.67, 308.95 ,207.64 ,183.17 , 97.25 ,147.03 ,236.14  , 0.   , 74.63],
                                [584.59 , 74.33, 126.29 ,194.49 ,179.61 ,363.58 ,275.86 ,353.28 ,140.43 ,145.99,
                                122.26 ,287.38, 244.07 ,259.85 , 72.24 ,330.44 ,212.02 ,116.62 ,305.45 ,128.41,
                                320.69 ,156.43 ,296.28 ,144.35 ,256.7  ,168.01 ,137.31 ,188.38 , 74.63 ,  0.  ],
                                ])
    data['demands'] =[0, 23, 19, 31, 40, 16, 36, 23, 12, 36, 18, 21, 10, 21, 18, 38, 21, 17, 29, 35, 31, 19, 18, 34, 14, 23, 10, 26, 35, 13]
    global total_demand_per_day
    total_demand_per_day=sum(data['demands'])
    global number_of_nodes
    number_of_nodes=len(data['demands'])

    data['num_vehicles'] = 3
    global number_of_routes_created
    number_of_routes_created=data['num_vehicles']
        
    data['vehicle_capacities'] = [150,150, 150]
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