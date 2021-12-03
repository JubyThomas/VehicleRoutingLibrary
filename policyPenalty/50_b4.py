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
mandatoryNodes=[]
mandatoryNodesById=[]
# [START data_model]



def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] =np.array([
        [0.0, 475.0, 353.23, 238.5, 282.85, 212.87, 255.1, 502.83, 256.63, 345.79, 282.39, 556.94, 378.89, 482.68, 408.04, 242.26, 308.17, 362.41, 495.8,
  214.39, 146.49, 602.42, 149.46, 553.91, 384.02, 482.28, 502.22, 321.59, 407.33, 512.56, 545.62, 440.22, 451.47, 637.36, 227.16, 451.22, 440.78, 584.99, 
  134.47, 620.46, 591.76, 464.26, 144.03, 565.91, 396.36, 327.15, 160.55, 440.7, 130.7, 268.94], [475.0, 0.0, 130.03, 305.28, 196.47, 289.0, 237.23, 
 30.27, 264.33, 429.31, 469.31, 310.2, 96.57, 56.65, 261.47, 232.86, 487.02, 302.17, 422.09, 373.8, 364.5, 342.26, 326.68, 129.56, 176.78, 
 268.09, 107.84, 528.6, 265.84, 122.54, 182.21, 37.2, 225.44, 406.99, 250.89, 233.55, 60.17, 395.25, 432.49, 376.65, 313.46, 57.04, 333.64, 
 148.6, 224.54, 178.24, 418.49, 190.95, 361.45, 262.61], [353.23, 130.03, 0.0, 228.12, 70.38, 159.05, 147.25, 160.15, 135.82, 386.17, 403.3,
  366.66, 55.17, 160.11, 263.88, 113.99, 425.8, 279.09, 429.33, 293.47, 235.03, 407.6, 209.58, 243.26, 173.62, 303.66, 202.05, 465.24, 267.29,
  217.12, 272.41, 103.08, 255.81, 467.33, 126.09, 262.09, 123.07, 440.07, 332.34, 439.19, 382.83, 145.0, 209.5, 260.65, 225.68, 126.0, 324.21, 
  222.64, 251.91, 136.01], [238.5, 305.28, 228.12, 0.0, 182.02, 211.01, 81.6, 325.83, 240.42, 173.05, 175.31, 318.45, 217.41, 289.83, 169.54, 
143.03, 198.6, 127.88, 279.26, 68.59, 228.81, 364.0, 126.37, 344.14, 160.59, 243.82, 292.5, 237.32, 168.87, 299.15, 320.04, 268.26, 215.13, 401.17, 
160.13, 214.11, 253.07, 351.28, 140.04, 382.9, 353.39, 272.43, 164.4, 352.1, 160.38, 127.28, 119.27, 209.4, 110.46, 253.78], 
  [282.85, 196.47, 70.38,182.02, 0.0, 96.77, 102.18, 225.89, 89.27, 351.28, 355.36, 386.82, 105.57, 216.95, 262.83, 46.86, 379.82, 262.08, 420.43,
   241.36, 169.07, 430.73, 140.43, 297.74, 183.45, 316.76, 251.15, 417.23, 265.36, 265.0, 314.45, 165.01, 270.48, 485.82, 55.71, 275.18, 175.93, 
   451.05, 268.63, 459.73, 409.03, 199.74, 139.18, 313.6, 228.54, 118.98, 263.49, 241.93, 185.27, 96.43], [212.87, 289.0, 159.05, 211.01, 96.77, 
 0.0, 152.87, 319.01, 45.0, 382.46, 364.05, 471.69, 201.88, 313.34, 336.5, 87.05, 391.0, 321.22, 481.01, 250.29, 77.13, 516.78, 106.89, 
 394.49, 268.8, 398.63, 347.8, 422.93, 338.2, 361.51, 409.69, 259.85, 354.68, 568.29, 60.11, 358.26, 272.61, 528.61, 243.18, 543.85,
 497.08, 296.34, 79.18, 410.37, 306.94, 201.21, 248.29, 329.58, 157.65, 58.83], [255.1, 237.23, 147.25, 81.6, 102.18, 152.87, 0.0, 261.24, 
  172.17, 249.13, 256.82, 326.8, 143.53, 
232.74, 185.35, 70.23, 280.18, 169.07, 329.16, 146.29, 194.98, 372.7, 109.84, 299.55, 132.7, 251.85, 247.59, 318.85, 186.68, 257.56, 
291.87, 200.14, 211.59, 420.12, 94.02, 213.71, 192.2, 377.64, 195.27, 397.34, 356.06, 214.41, 136.57, 311.01, 159.48, 72.06, 182.33, 
192.92, 129.07, 184.05], [502.83, 30.27, 160.15, 325.83, 225.89, 319.01, 261.24, 0.0, 294.59, 441.32, 485.74, 300.39, 123.94, 50.01, 
                          266.87, 261.08, 502.3, 312.08, 422.83, 394.03, 394.29, 329.42, 354.01, 106.04, 187.77, 265.0, 95.19, 544.03, 
                          271.32, 108.17, 165.85, 62.64, 225.34, 394.62, 279.87, 233.57, 73.54, 387.07, 456.08, 364.08, 300.0, 59.94,
                          362.25, 125.1, 231.74, 198.6, 441.02, 192.35, 387.23, 292.84],
[256.63, 264.33, 135.82, 240.42, 89.27, 45.0, 172.17, 294.59, 0.0, 413.41, 401.36, 475.88, 185.7, 295.41, 348.37, 101.95, 427.85,
 340.65, 500.64, 286.39, 115.41, 519.94, 148.63, 378.22, 272.28, 405.18, 334.57, 461.29, 350.57, 349.08, 401.04, 238.9, 359.31, 
 574.56, 80.41, 363.79, 256.67, 538.52, 286.18, 548.73, 498.3, 279.57, 123.82, 395.14, 315.65, 206.53, 289.7, 331.17, 199.86, 14.56],
[345.79, 429.31, 386.17, 173.05, 351.28, 382.46, 249.13, 441.32, 413.41, 0.0, 86.58, 293.48, 360.22, 395.13, 187.97, 315.41, 81.79, 
 131.14, 164.01, 141.42, 388.28, 329.29, 286.99, 419.49, 253.56, 241.05, 375.83, 118.19, 183.86, 375.47, 368.05, 395.79, 246.02, 338.18, 
 333.17, 239.61, 369.77, 279.35, 211.62, 332.54, 333.49, 381.63, 325.74, 419.76, 214.01, 264.39, 185.26, 265.19, 250.53, 426.82], [282.39,
 469.31, 403.3, 175.31, 355.36, 364.05, 256.82, 485.74, 401.36, 86.58, 0.0, 375.49, 388.49, 443.36, 254.92, 312.93, 27.66, 193.66,
 249.9, 115.41, 352.81, 413.24, 259.38, 479.76, 301.5, 317.2, 432.31, 62.03, 251.38, 434.44, 435.77, 433.44, 313.94, 424.52, 324.57,
 308.65, 412.32, 365.79, 150.3, 418.1, 415.46, 427.87, 295.63, 482.61, 272.64, 293.46, 129.1, 326.8, 213.41, 415.65], [556.94, 310.2, 
366.66, 318.45, 386.82, 471.69, 326.8, 300.39, 475.88, 293.48, 375.49, 0.0, 313.02, 254.03, 148.95, 384.81, 375.2, 199.57, 172.63, 355.53,
521.77, 46.07, 430.68, 215.03, 203.73, 75.58, 205.2, 410.55, 149.62, 192.94, 142.69, 298.1, 117.03, 100.7, 411.99, 113.6, 263.5, 88.14, 445.38, 
73.0, 40.02, 254.69, 462.28, 202.87, 167.39, 270.74, 418.9, 145.45, 428.03, 483.1], [378.89, 96.57, 55.17, 217.41, 105.57, 201.88, 143.53, 123.94, 
 185.7, 360.22, 388.49, 313.02, 0.0, 111.46, 219.32, 137.3, 408.72, 243.25, 385.98, 285.7, 274.47, 353.28, 230.22, 193.13, 127.14, 252.38, 148.98, 
 449.53, 223.09, 163.68, 217.75, 61.39, 204.64, 413.72, 157.26, 211.38, 71.17, 388.48, 338.28, 385.24, 328.12, 94.59, 238.73, 209.66, 180.35, 96.25,
 325.85, 170.49, 265.2, 187.83], [482.68, 56.65, 160.11, 289.83, 216.95, 313.34, 232.74, 50.01, 295.41, 395.13, 443.36, 254.03, 111.46, 0.0, 217.71, 
                                  245.1, 458.7, 264.91, 372.82, 356.95, 385.5, 285.62, 333.24, 83.26, 142.52, 215.39, 51.48, 500.47, 222.17, 66.57, 126.48, 58.83, 175.33, 350.39, 267.15, 183.56, 42.11, 339.7, 424.58, 320.03, 256.82, 18.44, 346.47, 101.12, 183.48, 164.62, 407.53, 142.44, 361.4, 296.12], [408.04, 261.47, 263.88, 169.54, 262.83, 336.5, 185.35, 266.87, 348.37, 187.97, 254.92, 148.95, 219.32, 217.71, 0.0, 250.06, 262.77, 61.47, 166.68, 211.01, 378.89, 194.64, 284.0, 231.95, 92.27, 74.53, 190.65, 303.15, 4.47, 188.73, 181.28, 233.79, 59.06, 236.04, 276.48, 53.85, 201.85, 192.33, 299.74, 215.24, 184.18, 207.46, 317.37, 231.8, 39.05, 144.46, 274.15, 78.6, 279.15, 357.87], [242.26, 232.86, 113.99, 143.03, 46.86, 87.05, 70.23, 261.08, 101.95, 315.41, 312.93, 384.81, 137.3, 245.1, 250.06, 0.0, 338.11, 238.87, 398.94, 197.95, 144.92, 430.03, 95.77, 322.13, 182.5, 311.59, 272.66, 374.48, 251.91, 285.24, 329.24, 198.67, 267.85, 481.24, 27.2, 271.31, 203.04, 441.89, 221.77, 456.86, 410.79, 227.02, 101.6, 336.51, 219.92, 114.83, 216.98, 243.59, 138.71, 113.89], [308.17, 487.02, 425.8, 198.6, 379.82, 391.0, 280.18, 502.3, 427.85, 81.79, 27.66, 375.2, 408.72, 458.7, 262.77, 338.11, 0.0, 202.51, 239.53, 141.51, 380.47, 411.0, 
286.7, 491.17, 316.25, 320.92, 444.96, 41.77, 258.97, 446.22, 444.04, 451.64, 321.74, 418.04, 350.53, 315.93, 429.17, 359.01, 177.13, 413.57, 415.22, 443.74, 323.13, 493.04, 283.77, 312.99, 156.54, 337.2, 241.01, 442.09], [362.41, 302.17, 279.09, 127.88, 262.08, 321.22, 169.07, 312.08, 340.65, 131.14, 193.66, 199.57, 243.25, 264.91, 61.47, 238.87, 202.51, 0.0, 160.1, 156.02, 352.32, 243.39, 251.98, 289.28, 125.4, 130.25, 244.76, 243.4, 58.19, 244.81, 242.14, 270.49, 120.34, 274.96, 263.06, 115.32, 242.08, 223.65, 245.97, 258.68, 237.54, 252.16, 288.5, 290.6, 82.88, 153.12, 219.35, 134.95, 237.7, 351.92], [495.8, 422.09, 429.33, 279.26, 420.43, 481.01, 329.16, 422.83, 500.64, 164.01, 249.9, 172.63, 385.98, 372.82, 166.68, 398.94, 239.53, 160.1, 0.0, 281.6, 507.91, 193.13, 405.63, 363.09, 258.94, 161.76, 335.05, 266.03, 162.89, 327.95, 295.3, 397.56, 197.5, 183.0, 423.13, 189.26, 364.16, 124.66, 365.47, 185.05, 208.22, 365.86, 443.54, 355.98, 205.7, 306.1, 337.85, 231.22, 382.62, 511.74], [214.39, 373.8, 293.47, 68.59, 241.36, 250.29, 146.29, 394.03, 286.39, 141.42, 115.41, 355.53, 285.7, 356.95, 211.01, 197.95, 141.51, 156.02, 281.6, 0.0, 247.78, 399.36, 148.49, 407.46, 222.97, 284.51, 356.5, 176.59, 209.09, 362.01, 377.39, 336.82, 264.28, 427.86, 209.16, 261.62, 321.01, 373.43, 90.03, 413.55, 393.14, 339.82, 186.55, 414.12, 212.25, 195.67, 63.39, 265.11, 109.13, 300.6], [146.49, 364.5, 235.03, 228.81, 169.07, 77.13, 194.98, 394.29, 115.41, 388.28, 352.81, 521.77, 274.47, 385.5, 378.89, 144.92, 380.47, 352.32, 507.91, 247.78, 0.0, 567.63, 102.93, 465.0, 323.8, 446.81, 416.56, 406.72, 379.82, 429.53, 474.14, 334.01, 406.12, 614.6, 119.64, 408.53, 344.0, 570.31, 213.0, 592.23, 550.47, 367.97, 64.4, 480.06, 354.46, 256.98, 226.28, 385.05, 139.98, 125.94], [602.42, 342.26, 407.6, 364.0, 430.73, 516.78, 372.7, 329.42, 519.94, 329.29, 413.24, 46.07, 353.28, 285.62, 194.64, 430.03, 411.0, 243.39, 193.13, 399.36, 567.63, 0.0, 476.74, 235.85, 248.28, 121.64, 234.98, 444.62, 195.14, 221.29, 165.22, 333.64, 162.2, 65.31, 457.22, 159.2, 299.67, 81.27, 489.35, 34.71, 30.07, 288.48, 508.3, 221.11, 213.35, 
315.64, 462.62, 188.81, 473.79, 526.8], [149.46, 326.68, 209.58, 126.37, 140.43, 106.89, 109.84, 354.01, 148.63, 286.99, 259.38, 430.68, 230.22, 333.24, 284.0, 95.77, 286.7, 251.98, 405.63, 148.49, 102.93, 476.74, 0.0, 405.45, 242.32, 355.11, 354.09, 317.1, 284.4, 365.01, 401.51, 291.37, 318.1, 519.85, 
87.68, 319.34, 291.32, 473.0, 137.88, 499.12, 462.09, 314.84, 38.91, 418.03, 264.09, 180.8, 141.44, 301.92, 51.26, 163.17], [553.91, 129.56, 243.26, 344.14, 297.74, 394.49, 299.55, 106.04, 378.22, 419.49, 479.76, 215.03, 193.13, 83.26, 231.95, 322.13, 491.17, 289.28, 363.09, 407.46, 465.0, 235.85, 405.45, 0.0, 184.49, 201.7, 52.17, 532.41, 236.21, 45.65, 72.34, 141.78, 176.48, 300.82, 345.68, 184.01, 122.0, 303.12, 483.27, 270.35, 205.79, 98.67, 422.3, 19.1, 207.42, 227.86, 463.41, 154.38, 428.27, 379.23], [384.02, 176.78, 173.62, 160.59, 183.45, 268.8, 132.7, 187.77, 272.28, 253.56, 301.5, 203.73, 127.14, 142.52, 92.27, 182.5, 316.25, 125.4, 258.94, 222.97, 323.8, 248.28, 242.32, 184.49, 0.0, 133.6, 133.76, 358.02, 96.13, 139.09, 162.01, 145.5, 87.05, 302.37, 209.68, 91.83, 116.73, 269.13, 300.51, 276.48, 228.32, 128.22, 268.53, 191.64, 53.24, 67.68, 279.6, 61.29, 254.1, 279.87], [482.28, 268.09, 303.66, 243.82, 316.76, 398.63, 251.85, 265.0, 405.18, 241.05, 317.2, 75.58, 252.38, 215.39, 74.53, 311.59, 320.92, 130.25, 161.76, 284.51, 446.81, 121.64, 355.11, 201.7, 133.6, 0.0, 174.2, 359.17, 75.8, 166.38, 136.4, 248.77, 47.85, 169.73, 338.66, 41.88, 213.97, 136.42, 373.74, 145.6, 109.66, 211.06, 386.91, 195.69, 92.42, 198.8, 347.83, 82.22, 352.92, 413.18], [502.22, 107.84, 202.05, 292.5, 251.15, 347.8, 247.59, 95.19, 334.57, 375.83, 432.31, 205.2, 148.98, 51.48, 190.65, 272.66, 444.96, 244.76, 335.05, 356.5, 416.56, 234.98, 354.09, 52.17, 133.76, 174.2, 0.0, 486.49, 195.06, 15.56, 75.21, 106.78, 140.04, 300.0, 296.95, 148.14, 79.08, 292.07, 431.34, 269.53, 205.91, 59.41, 371.98, 64.07, 161.94, 175.79, 411.75, 112.07, 376.19, 336.78], [321.59, 528.6, 465.24, 237.32, 417.23, 422.93, 318.85, 544.03, 461.29, 118.19, 62.03, 410.55, 449.53, 500.47, 303.15, 374.48, 41.77, 243.4, 266.03, 176.59, 406.72, 444.62, 317.1, 532.41, 358.02, 359.17, 486.49, 0.0, 299.26, 487.57, 484.29, 493.12, 361.96, 447.53, 385.41, 355.95, 470.87, 388.49, 196.94, 445.07, 450.47, 485.51, 352.12, 534.0, 325.09, 354.04, 180.47, 378.28, 269.19, 475.67], [407.33, 265.84, 267.29, 168.87, 265.36, 338.2, 186.68, 271.32, 350.57, 183.86, 251.38, 149.62, 223.09, 222.17, 4.47, 251.91, 258.97, 58.19, 162.89, 209.09, 379.82, 195.14, 284.4, 236.21, 96.13, 75.8, 195.06, 299.26, 0.0, 193.08, 185.08, 238.05, 62.77, 235.48, 278.23, 57.27, 206.2, 190.97, 298.05, 215.13, 185.33, 211.9, 318.06, 235.93, 43.0, 147.23, 272.31, 83.01, 278.73, 360.19], [512.56, 122.54, 217.12, 299.15, 265.0, 361.51, 257.56, 108.17, 349.08, 375.47, 434.44, 192.94, 163.68, 66.57, 188.73, 285.24, 446.22, 244.81, 327.95, 362.01, 429.53, 221.29, 365.01, 45.65, 139.09, 166.38, 15.56, 487.57, 193.08, 0.0, 59.91, 122.33, 135.48, 286.46, 309.89, 143.39, 94.3, 280.35, 438.6, 255.93, 191.97, 74.97, 383.89, 53.6, 162.48, 185.53, 418.38, 110.39, 385.63, 351.51], [545.62, 182.21, 272.41, 320.04, 314.45, 409.69, 291.87, 165.85, 401.04, 368.05, 435.77, 142.69, 217.75, 126.48, 181.28, 329.24, 444.04, 242.14, 295.3, 377.39, 474.14, 165.22, 401.51, 72.34, 162.01, 136.4, 75.21, 
484.29, 185.08, 59.91, 0.0, 181.47, 122.33, 230.5, 355.14, 128.46, 151.43, 230.78, 459.83, 199.91, 135.33, 134.06, 424.23, 61.35, 166.36, 220.95, 437.34, 112.36, 416.03, 404.57], [440.22, 37.2, 103.08, 268.26, 165.01, 259.85, 200.14, 62.64, 238.9, 395.79, 433.44, 298.1, 61.39, 58.83, 233.79, 198.67, 451.64, 270.49, 397.56, 336.82, 334.01, 333.64, 291.37, 141.78, 145.5, 248.77, 106.78, 493.12, 238.05, 122.33, 181.47, 0.0, 203.63, 397.21, 218.11, 211.46, 34.93, 380.38, 395.38, 367.38, 306.0, 47.42, 300.03, 159.88, 195.72, 141.43, 381.29, 168.36, 325.01, 238.8], [451.47, 225.44, 255.81, 215.13, 270.48, 354.68, 211.59, 225.34, 359.31, 246.02, 313.94, 117.03, 204.64, 175.33, 59.06, 267.85, 321.74, 120.34, 197.5, 264.28, 406.12, 162.2, 318.1, 176.48, 87.05, 47.85, 140.04, 361.96, 62.77, 135.48, 122.33, 203.63, 0.0, 215.34, 295.04, 8.25, 169.19, 184.26, 351.07, 189.55, 144.47, 168.76, 347.94, 174.66, 55.11, 153.72, 326.55, 35.36, 320.95, 366.9], [637.36, 406.99, 467.33, 401.17, 485.82, 568.29, 420.12, 394.62, 574.56, 338.18, 424.52, 100.7, 413.72, 350.39, 236.04, 481.24, 418.04, 274.96, 183.0, 427.86, 614.6, 65.31, 519.85, 300.82, 302.37, 169.73, 300.0, 447.53, 235.48, 286.46, 230.5, 397.21, 215.34, 0.0, 508.27, 210.79, 362.91, 59.08, 517.45, 30.59, 95.19, 352.66, 553.4, 285.7, 261.52, 368.42, 490.03, 245.6, 511.62, 582.23], [227.16, 250.89, 126.09, 160.13, 55.71, 60.11, 94.02, 279.87, 80.41, 333.17, 324.57, 411.99, 157.26, 267.15, 276.48, 27.2, 350.53, 263.06, 423.13, 209.16, 119.64, 457.22, 87.68, 345.68, 209.68, 338.66, 296.95, 385.41, 278.23, 309.89, 355.14, 218.11, 295.04, 508.27, 0.0, 298.47, 225.32, 468.52, 221.96, 483.99, 437.98, 249.34, 83.49, 360.53, 246.86, 142.0, 220.57, 270.74, 136.01, 93.65], [451.22, 233.55, 262.09, 214.11, 275.18, 358.26, 213.71, 233.57, 363.79, 239.61, 308.65, 113.6, 211.38, 183.56, 53.85, 271.31, 315.93, 115.32, 189.26, 261.62, 408.53, 159.2, 319.34, 184.01, 91.83, 41.88, 148.14, 355.95, 57.27, 143.39, 128.46, 211.46, 8.25, 210.79, 298.47, 0.0, 177.1, 178.13, 349.0, 185.61, 142.76, 176.92, 349.78, 181.79, 55.36, 157.7, 324.19, 43.1, 320.88, 371.6], [440.78, 60.17, 123.07, 253.07, 175.93, 272.61, 192.2, 73.54, 256.67, 369.77, 412.32, 263.5, 71.17, 42.11, 201.85, 203.04, 429.17, 242.08, 364.16, 321.01, 344.0, 299.67, 291.32, 122.0, 116.73, 213.97, 79.08, 470.87, 206.2, 94.3, 151.43, 34.93, 169.19, 362.91, 225.32, 177.1, 0.0, 345.48, 385.52, 333.2, 272.38, 24.02, 304.37, 138.51, 164.56, 126.25, 369.42, 134.03, 320.36, 258.3], [584.99, 395.25, 440.07, 351.28, 451.05, 528.61, 377.64, 387.07, 538.52, 279.35, 365.79, 
88.14, 388.48, 339.7, 192.33, 441.89, 359.01, 223.65, 124.66, 373.43, 570.31, 81.27, 473.0, 303.12, 269.13, 136.42, 292.07, 388.49, 190.97, 280.35, 230.78, 380.38, 184.26, 59.08, 468.52, 178.13, 345.48, 0.0, 462.47, 62.43, 106.51, 338.89, 507.88, 290.63, 222.87, 332.2, 434.89, 218.0, 461.34, 547.2], [134.47, 432.49, 332.34, 140.04, 268.63, 243.18, 195.27, 456.08, 286.18, 211.62, 150.3, 445.38, 338.28, 424.58, 299.74, 221.77, 177.13, 245.97, 365.47, 90.03, 213.0, 489.35, 137.88, 483.27, 300.51, 373.74, 431.34, 196.94, 298.05, 438.6, 459.83, 395.38, 351.07, 517.45, 221.96, 349.0, 385.52, 462.47, 0.0, 503.51, 482.74, 406.59, 165.66, 491.77, 297.54, 259.96, 27.73, 348.54, 86.76, 300.68], [620.46, 376.65, 439.19, 382.9, 459.73, 543.85, 397.34, 364.08, 548.73, 332.54, 418.1, 73.0, 385.24, 320.03, 215.24, 456.86, 413.57, 258.68, 185.05, 413.55, 592.23, 34.71, 499.12, 270.35, 276.48, 145.6, 269.53, 445.07, 
215.13, 255.93, 199.91, 367.38, 189.55, 30.59, 483.99, 185.61, 333.2, 62.43, 503.51, 0.0, 64.63, 322.56, 531.81, 255.38, 238.0, 343.21, 476.35, 218.45, 
493.26, 556.04], [591.76, 313.46, 382.83, 353.39, 409.03, 497.08, 356.06, 300.0, 498.3, 333.49, 415.46, 40.02, 328.12, 256.82, 184.18, 410.79, 415.22, 237.54, 208.22, 393.14, 550.47, 30.07, 462.09, 205.79, 228.32, 109.66, 205.91, 450.47, 185.33, 191.97, 135.33, 306.0, 144.47, 95.19, 437.98, 142.76, 272.38, 106.51, 482.74, 64.63, 0.0, 260.32, 492.38, 191.07, 198.01, 295.99, 456.53, 167.61, 462.09, 504.64], [464.26, 57.04, 145.0, 272.43, 199.74, 296.34, 
214.41, 59.94, 279.57, 381.63, 427.87, 254.69, 94.59, 18.44, 207.46, 227.02, 443.74, 252.16, 365.86, 339.82, 367.97, 288.48, 314.84, 98.67, 128.22, 211.06, 59.41, 485.51, 211.9, 74.97, 134.06, 47.42, 168.76, 352.66, 249.34, 176.92, 24.02, 338.89, 406.59, 322.56, 260.32, 0.0, 328.3, 115.69, 171.9, 146.69, 389.79, 134.65, 343.0, 280.78], [144.03, 333.64, 209.5, 164.4, 139.18, 79.18, 136.57, 362.25, 123.82, 325.74, 295.63, 462.28, 238.73, 346.47, 317.37, 
101.6, 323.13, 288.5, 443.54, 186.55, 64.4, 508.3, 38.91, 422.3, 268.53, 386.91, 371.98, 352.12, 318.06, 383.89, 424.23, 300.03,
347.94, 553.4, 83.49, 349.78, 304.37, 507.88, 165.66, 531.81, 492.38, 328.3, 0.0, 436.03, 294.98, 203.75, 173.57, 329.26, 83.01, 137.93],
[565.91, 148.6, 260.65, 352.1, 313.6, 410.37, 311.01, 125.1, 395.14, 419.76, 482.61, 202.87, 209.66, 101.12, 231.8, 336.51, 493.04, 290.6,
 355.98, 414.12, 480.06, 221.11, 418.03, 19.1, 191.64, 195.69, 64.07, 534.0, 235.93, 53.6, 61.35, 159.88, 174.66, 285.7, 360.53, 181.79, 138.51,
 290.63, 491.77, 255.38, 191.07, 115.69, 436.03, 0.0, 210.0, 239.01, 471.22, 155.9, 439.21, 396.47], [396.36, 224.54, 225.68, 160.38, 228.54, 306.94, 
 159.48, 231.74, 315.65, 214.01, 272.64, 167.39, 180.35, 183.48, 39.05, 219.92, 283.77, 82.88, 205.7, 212.25, 354.46, 213.35, 264.09, 207.42,
 53.24, 92.42, 161.94, 325.09, 43.0, 162.48, 166.36, 195.72, 55.11, 261.52, 246.86, 55.36, 164.56, 222.87, 297.54, 238.0, 198.01, 171.9, 294.98,
 210.0, 0.0, 109.55, 273.65, 54.56, 265.86, 324.45], [327.15, 178.24, 126.0, 127.28, 118.98, 201.21, 72.06, 198.6, 206.53, 264.39, 293.46, 270.74, 
                                                      96.25, 164.62, 144.46, 114.83, 312.99, 153.12, 306.1, 195.67, 256.98, 315.64, 180.8, 227.86, 
                                                      67.68, 198.8, 175.79, 354.04, 147.23, 185.53, 220.95, 141.43, 153.72, 368.42, 142.0, 157.7, 
                                                      126.25, 332.2, 259.96, 343.21, 295.99, 146.69, 203.75, 239.01, 109.55, 0.0, 243.23, 128.8, 200.41, 215.0],
 [160.55, 418.49, 324.21, 119.27, 263.49, 248.29, 182.33, 441.02, 289.7, 185.26, 129.1, 418.9, 325.85, 407.53, 274.15, 216.98, 156.54, 219.35,
  337.85, 63.39, 226.28, 462.62, 141.44, 463.41, 279.6, 347.83, 411.75, 180.47, 272.31, 418.38, 437.34, 381.29, 326.55, 490.03, 220.57, 324.19, 
  369.42, 434.89, 27.73, 476.35, 456.53, 389.79, 173.57, 471.22, 273.65, 243.23, 0.0, 325.51, 90.93, 304.26],
 [440.7, 190.95, 222.64, 209.4, 241.93, 329.58, 192.92, 192.35, 331.17, 265.19, 326.8, 145.45, 170.49, 142.44, 78.6, 
  243.59, 337.2, 134.95, 231.22, 265.11, 385.05, 188.81, 301.92, 154.38, 61.29, 82.22, 112.07, 378.28, 83.01, 110.39,
  112.36, 168.36, 35.36, 245.6, 270.74, 43.1, 134.03, 218.0, 348.54, 218.45, 167.61, 134.65, 329.26, 155.9, 54.56, 128.8,
  325.51, 0.0, 310.1, 338.0], [130.7, 361.45, 251.91, 110.46, 185.27, 157.65, 129.07, 387.23, 199.86, 250.53, 213.41, 428.03, 265.2,
                               361.4, 279.15, 138.71, 241.01, 237.7, 382.62, 109.13, 139.98, 473.79, 51.26, 428.27, 254.1, 352.92, 
                               376.19, 269.19, 278.73, 385.63, 416.03, 325.01, 320.95, 511.62, 136.01, 320.88, 320.36, 461.34, 86.76,
                               493.26, 462.09, 343.0, 83.01, 439.21, 265.86, 200.41, 90.93, 310.1, 0.0, 214.41],
  [268.94, 262.61, 136.01, 253.78, 96.43, 58.83, 184.05, 292.84, 14.56, 426.82, 415.65, 483.1, 187.83, 296.12, 357.87, 
   113.89, 442.09, 351.92, 511.74, 300.6, 125.94, 526.8, 163.17, 379.23, 279.87, 413.18, 336.78, 475.67, 360.19, 351.51, 404.57,
   238.8, 366.9, 582.23, 93.65, 371.6, 258.3, 547.2, 300.68, 556.04, 504.64, 280.78, 137.93, 396.47, 324.45, 215.0, 304.26, 338.0, 214.41, 0.0]
   

                         ])     
                             
                            


    #"""To form the the Demand"""
    # if count ==1:
    data['demands'] =[0, 23, 39, 25, 18, 34, 6, 20, 28, 4, 42, 19, 33, 39, 39, 18, 24, 32, 18, 44, 12, 12, 33, 11, 19, 28, 30, 35, 38,17,
                      45, 18, 36, 30, 29, 18, 24, 10, 37, 13, 15, 24, 42, 7, 21, 39, 19, 23, 25, 37]


    data['num_vehicles'] = 4
    data['vehicle_capacities'] = [200,200,200,200]


    data['depot'] = 0

    dictSort={}

    for val in range(len(data['demands'])):
        dictSort[val]=data['demands'][val]
 
    print(dictSort)

    #Creating Array which contains details of mandatory nodes to be picked up
    y= sorted(data["demands"],reverse=True)
    print(y)

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

    #need to drop Id =0 which is the depot since itcan notbe included in AddDisjunction method
    mandatoryNodesById.remove(0)


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