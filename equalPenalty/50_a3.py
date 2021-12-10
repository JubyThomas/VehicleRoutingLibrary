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
        
        [0.0, 418.57, 637.91, 355.01, 671.3, 586.76, 196.35, 175.86, 419.51, 252.38, 235.7, 256.91, 411.54, 457.14, 501.94, 454.53, 731.14, 365.31, 729.51, 352.9, 475.46, 335.83, 536.27, 621.1, 426.97, 411.34, 385.62, 469.26, 318.66, 640.19, 532.92, 639.46, 303.49, 608.84, 548.0, 506.43, 488.44, 395.41, 304.72, 655.85, 441.99, 554.48, 553.09, 531.52, 280.44, 234.2, 543.41, 314.92, 432.39, 661.25], [418.57, 0.0, 348.17, 137.77, 275.19, 257.51, 305.6, 275.09, 171.64, 176.71, 223.91, 162.0, 329.25, 258.04, 128.84, 387.0, 345.74, 88.81, 376.43, 161.55, 128.28, 114.11, 346.3, 439.24, 210.41, 53.71, 81.58, 50.8, 118.47, 237.93, 317.35, 377.51, 115.11, 195.53, 152.61, 90.87, 174.4, 32.02, 341.6, 244.75, 195.24, 186.07, 389.67, 346.72, 219.18, 196.82, 176.28, 345.68, 220.06, 272.42], [637.91, 348.17, 0.0, 300.13, 213.51, 96.26, 448.32, 462.33, 226.61, 409.32, 404.68, 442.11, 269.41, 183.88, 433.34, 276.03, 
201.01, 319.63, 133.79, 292.96, 452.92, 336.05, 143.03, 160.9, 211.4, 401.05, 429.3, 335.86, 445.62, 230.1, 122.43, 45.54, 404.45, 273.64, 230.23, 350.6, 500.73, 338.64, 377.66, 267.02, 199.91, 177.86, 170.79, 149.38, 357.66, 420.86, 183.38, 371.71, 205.55, 191.48], [355.01, 137.77, 300.13, 0.0, 321.03, 234.44, 194.7, 183.74, 74.01, 110.11, 122.49, 143.89, 196.37, 149.97, 266.56, 256.22, 377.0, 50.16, 376.09, 24.17, 262.94, 42.06, 245.94, 343.24, 101.51, 178.44, 187.2, 175.01, 168.12, 294.11, 225.28, 314.46, 114.24, 275.18, 203.91, 217.06, 305.9, 106.61, 206.0, 316.46, 102.31, 200.89, 283.32, 244.31, 94.37, 124.06, 190.03, 209.29, 111.07, 308.85], [671.3, 275.19, 213.51, 321.03, 0.0, 146.17, 513.37, 504.39, 274.58, 419.0, 443.4, 425.94, 408.71, 301.73, 295.63, 440.6, 74.0, 306.22, 138.97, 330.29, 325.44, 335.48, 324.0, 371.81, 291.14, 313.25, 347.63, 234.83, 393.61, 40.0, 291.21, 259.05, 378.61, 97.16, 125.3, 222.23, 366.03, 286.69, 488.43, 62.43, 266.18, 122.2, 364.61, 329.04, 411.3, 437.14, 132.28, 486.57, 293.06, 22.36], [586.76, 257.51, 96.26, 234.44, 146.17, 0.0, 408.21, 411.83, 167.45, 344.06, 351.13, 368.77, 269.39, 164.48, 337.19, 296.15, 166.96, 242.54, 142.89, 233.9, 357.34, 264.14, 178.1, 235.46, 168.44, 309.26, 339.09, 241.08, 362.11, 
149.68, 145.06, 135.79, 326.99, 183.96, 134.18, 254.4, 404.95, 251.88, 361.01, 187.24, 147.8, 81.61, 219.79, 183.01, 309.32, 358.5, 87.32, 357.46, 166.9, 124.49], [196.35, 305.6, 448.32, 194.7, 513.37, 408.21, 0.0, 53.6, 242.99, 136.0, 81.79, 181.41, 216.01, 265.0, 422.0, 262.88, 564.45, 225.85, 550.36, 183.1, 406.12, 195.36, 340.27, 424.92, 241.36, 323.3, 312.03, 353.28, 254.28, 488.21, 338.5, 446.22, 207.02, 469.4, 398.58, 395.49, 436.9, 275.0, 117.55, 511.16, 260.42, 391.57, 356.88, 335.41, 102.07, 118.27, 381.18, 127.95, 245.1, 499.61], [175.86, 275.09, 462.33, 183.74, 504.39, 411.83, 53.6, 0.0, 244.38, 98.68, 61.4, 135.03, 252.05, 283.01, 384.0, 304.01, 560.53, 204.02, 554.7, 178.63, 365.42, 173.0, 366.66, 456.25, 251.18, 285.59, 270.45, 324.78, 209.01, 476.07, 360.22, 465.32, 167.69, 451.4, 384.51, 365.95, 393.02, 246.42, 165.68, 495.76, 266.24, 384.63, 388.42, 362.4, 104.68, 78.75, 373.77, 175.79, 256.78, 492.56], [419.51, 171.64, 226.61, 74.01, 274.58, 167.45, 242.99, 244.38, 0.0, 182.72, 183.81, 217.66, 166.51, 87.0, 294.89, 219.92, 321.73, 105.1, 310.34, 67.36, 299.35, 112.97, 179.78, 276.14, 38.83, 222.1, 239.58, 191.71, 234.39, 254.47, 155.35, 240.53, 184.82, 249.6, 173.42, 228.66, 346.01, 147.09, 217.61, 283.02, 28.86, 152.64, 220.84, 179.18, 142.41, 194.81, 143.27, 217.33, 48.41, 259.01], [252.38, 176.71, 409.32, 110.11, 419.0, 344.06, 136.0, 98.68, 182.72, 0.0, 58.24, 55.32, 255.48, 246.5, 286.8, 315.16, 480.08, 112.93, 484.86, 117.07, 270.15, 83.76, 340.48, 436.57, 203.12, 188.1, 176.07, 226.77, 121.17, 387.87, 325.23, 421.1, 71.03, 358.42, 295.68, 267.57, 301.4, 148.86, 209.91, 404.49, 209.77, 303.56, 371.95, 337.66, 88.53, 20.12, 292.41, 217.75, 211.4, 409.4], [235.7, 223.91, 404.68, 122.49, 443.4, 351.13, 81.79, 61.4, 183.81, 58.24, 0.0, 110.94, 215.23, 228.97, 340.85, 272.38, 499.14, 146.37, 493.94, 117.48, 326.08, 115.41, 317.35, 410.31, 193.31, 242.3, 232.95, 271.97, 179.4, 415.65, 307.53, 410.18, 127.77, 392.76, 324.45, 314.03, 358.83, 193.54, 154.4, 436.25, 206.7, 323.34, 343.43, 313.64, 50.09, 45.49, 312.51, 162.89, 199.68, 431.34], [256.91, 162.0, 442.11, 143.89, 425.94, 368.77, 181.41, 135.03, 217.66, 55.32, 110.94, 0.0, 308.87, 290.39, 253.65, 369.01, 491.59, 126.61, 505.63, 157.65, 232.4, 106.08, 385.91, 482.89, 243.97, 157.58, 137.54, 212.51, 74.01, 391.69, 367.81, 458.19, 47.8, 355.18, 300.67, 249.61, 258.09, 141.21, 264.61, 403.45, 246.17, 317.77, 419.87, 383.6, 143.54, 65.8, 306.65, 272.66, 253.05, 419.04], [411.54, 329.25, 269.41, 196.37, 408.71, 269.39, 216.01, 252.05, 166.51, 255.48, 215.23, 308.87, 0.0, 107.0, 456.85, 60.81, 436.04, 245.93, 394.4, 172.53, 457.06, 234.27, 135.69, 210.15, 131.55, 374.09, 383.4, 356.4, 356.39, 398.9, 147.19, 253.07, 298.41, 406.65, 331.12, 394.68, 501.38, 300.13, 114.95, 432.38, 156.12, 296.59, 142.58, 129.77, 168.65, 254.12, 290.24, 106.93, 125.06, 388.96], [457.14, 258.04, 183.88, 149.97, 301.73, 164.48, 265.0, 283.01, 87.0, 246.5, 228.97, 290.39, 107.0, 0.0, 378.65, 146.67, 331.42, 189.99, 297.02, 132.85, 385.03, 191.93, 96.01, 193.31, 49.65, 309.03, 326.38, 273.89, 317.24, 292.31, 79.06, 182.39, 264.2, 303.06, 228.88, 307.75, 432.15, 234.08, 198.8, 326.34, 63.51, 190.76, 134.53, 94.37, 179.61, 253.57, 185.13, 194.37, 39.66, 282.12], [501.94, 128.84, 433.34, 266.56, 295.63, 337.19, 422.0, 384.0, 294.89, 286.8, 340.85, 253.65, 456.85, 378.65, 0.0, 513.56, 369.3, 217.15, 424.45, 290.39, 34.01, 239.88, 460.3, 
548.04, 333.54, 98.73, 117.0, 105.42, 183.98, 255.88, 428.69, 469.88, 216.36, 198.65, 205.27, 82.86, 70.83, 160.1, 468.96, 243.05, 315.15, 255.89, 505.22, 461.65, 343.98, 305.95, 249.95, 473.47, 342.71, 302.68], [454.53, 387.0, 276.03, 256.22, 440.6, 296.15, 262.88, 304.01, 219.92, 315.16, 272.38, 369.01, 60.81, 146.67, 513.56, 0.0, 459.19, 305.37, 407.59, 232.61, 515.17, 294.74, 133.14, 180.43, 182.49, 433.12, 443.42, 411.5, 417.2, 435.49, 157.08, 250.34, 359.22, 449.45, 375.54, 448.43, 560.12, 358.62, 150.0, 470.66, 204.94, 336.01, 119.62, 126.75, 227.48, 312.82, 331.07, 140.01, 174.45, 419.7], [731.14, 345.74, 201.01, 377.0, 74.0, 166.96, 564.45, 560.53, 321.73, 480.08, 499.14, 491.59, 436.04, 331.42, 369.3, 459.19, 0.0, 368.51, 86.45, 382.99, 399.36, 396.35, 332.57, 361.31, 331.1, 385.86, 420.12, 307.43, 463.7, 114.0, 303.16, 244.35, 444.93, 170.66, 193.54, 296.17, 439.52, 354.75, 526.77, 130.08, 307.94, 176.67, 367.84, 338.35, 462.93, 497.32, 187.77, 523.64, 331.01, 73.38], [365.31, 88.81, 319.63, 50.16, 306.22, 242.54, 225.85, 204.02, 105.1, 112.93, 146.37, 126.61, 245.93, 189.99, 217.15, 305.37, 368.51, 0.0, 379.12, 74.32, 212.81, 31.06, 284.54, 381.18, 140.35, 128.36, 138.85, 130.25, 129.69, 274.97, 260.38, 339.89, 85.01, 247.42, 182.78, 172.82, 255.81, 57.07, 252.87, 292.41, 133.42, 192.78, 324.52, 283.67, 132.67, 131.59, 181.6, 256.88, 150.34, 297.12], [729.51, 376.43, 133.79, 376.09, 138.97, 142.89, 550.36, 554.7, 310.34, 484.86, 493.94, 505.63, 394.4, 297.02, 424.45, 407.59, 86.45, 379.12, 0.0, 376.61, 451.47, 403.22, 275.54, 287.37, 309.4, 423.08, 456.0, 346.76, 489.94, 173.99, 251.08, 171.64, 461.57, 231.23, 227.5, 345.18, 495.26, 378.32, 495.8, 201.36, 290.11, 191.39, 304.53, 281.78, 452.1, 500.02, 201.95, 491.13, 306.73, 124.81], [352.9, 161.55, 292.96, 24.17, 330.29, 233.9, 183.1, 178.63, 67.36, 117.07, 117.48, 157.65, 172.53, 132.85, 290.39, 232.61, 382.99, 74.32, 376.61, 0.0, 287.09, 63.03, 228.65, 325.85, 86.7, 202.6, 210.87, 197.37, 188.81, 305.5, 210.24, 304.04, 133.09, 290.41, 217.19, 239.05, 330.05, 130.6, 184.22, 329.82, 93.06, 208.66, 264.33, 226.59, 81.06, 127.91, 198.17, 187.0, 95.52, 316.81], [475.46, 128.28, 452.92, 262.94, 325.44, 357.34, 406.12, 365.42, 299.35, 270.15, 326.08, 232.4, 457.06, 385.03, 34.01, 515.17, 399.36, 212.81, 451.47, 287.09, 0.0, 232.16, 470.38, 560.61, 338.18, 86.12, 95.04, 118.22, 160.16, 285.51, 439.84, 487.71, 199.12, 229.01, 228.37, 106.48, 48.41, 157.01, 460.76, 274.66, 321.72, 277.0, 514.66, 471.29, 333.9, 288.56, 270.12, 465.93, 
347.7, 331.3], [335.83, 114.11, 336.05, 42.06, 335.48, 264.14, 195.36, 173.0, 112.97, 83.76, 115.41, 106.08, 234.27, 191.93, 239.88, 294.74, 396.35, 31.06, 403.22, 63.03, 232.16, 0.0, 287.85, 385.13, 143.17, 146.16, 150.08, 158.9, 126.19, 305.0, 266.58, 352.94, 72.45, 278.4, 212.9, 201.43, 272.61, 82.22, 229.17, 323.14, 141.79, 219.91, 325.38, 286.3, 105.08, 101.67, 208.75, 233.98, 152.85, 325.65], [536.27, 346.3, 143.03, 245.94, 324.0, 178.1, 340.27, 366.66, 179.78, 340.48, 317.35, 385.91, 135.69, 96.01, 460.3, 133.14, 332.57, 284.54, 275.54, 228.65, 470.38, 287.85, 0.0, 97.31, 144.9, 398.84, 418.63, 354.92, 412.79, 326.34, 34.48, 118.93, 360.17, 352.06, 285.62, 383.98, 518.35, 325.04, 250.01, 363.46, 152.63, 237.19, 45.65, 6.4, 267.27, 346.08, 235.72, 242.42, 135.09, 302.04], [621.1, 439.24, 160.9, 343.24, 371.81, 235.46, 424.92, 456.25, 276.14, 436.57, 410.31, 482.89, 210.15, 193.31, 548.04, 180.43, 361.31, 381.18, 287.37, 325.85, 560.61, 385.13, 97.31, 0.0, 242.08, 492.4, 513.65, 443.4, 509.84, 383.36, 122.07, 117.02, 457.47, 419.05, 360.84, 468.96, 608.91, 419.65, 
323.12, 420.96, 248.34, 308.89, 68.07, 99.3, 360.35, 441.28, 310.08, 314.09, 232.31, 349.49], [426.97, 210.41, 211.4, 101.51, 291.14, 168.44, 241.36, 251.18, 38.83, 203.12, 193.31, 
243.97, 131.55, 49.65, 333.54, 182.49, 331.1, 140.35, 309.4, 86.7, 338.18, 143.17, 144.9, 242.08, 0.0, 260.53, 277.04, 229.93, 267.9, 275.47, 123.94, 218.59, 215.6, 277.22, 201.1, 266.05, 384.8, 185.35, 197.37, 306.62, 25.71, 172.07, 184.17, 143.66, 146.54, 212.29, 164.17, 195.45, 10.0, 273.62], [411.34, 53.71, 401.05, 178.44, 313.25, 309.26, 323.3, 285.59, 222.1, 188.1, 242.3, 157.58, 374.09, 309.03, 98.73, 433.12, 385.86, 128.36, 423.08, 202.6, 86.12, 146.16, 398.84, 492.4, 260.53, 0.0, 34.48, 78.45, 94.18, 274.31, 370.41, 431.02, 117.9, 225.44, 196.12, 101.51, 127.46, 75.29, 374.64, 275.02, 246.95, 235.05, 441.74, 399.06, 247.93, 207.33, 225.89, 379.82, 270.35, 313.25], [385.62, 81.58, 429.3, 187.2, 347.63, 339.09, 312.03, 270.45, 239.58, 176.07, 232.95, 137.54, 383.4, 326.38, 117.0, 443.42, 420.12, 138.85, 456.0, 210.87, 95.04, 150.08, 418.63, 513.65, 277.04, 34.48, 0.0, 112.81, 67.08, 308.76, 391.64, 457.56, 105.23, 259.83, 229.63, 134.83, 126.3, 94.15, 373.5, 309.38, 266.01, 266.83, 460.4, 418.4, 245.49, 194.04, 257.32, 379.49, 287.01, 347.38], [469.26, 50.8, 335.86, 175.01, 234.83, 241.08, 353.28, 324.78, 191.71, 226.77, 271.97, 212.51, 356.4, 273.89, 105.42, 411.5, 307.43, 130.25, 346.76, 197.37, 118.22, 158.9, 354.92, 443.4, 229.93, 78.45, 112.81, 0.0, 163.78, 196.02, 323.49, 369.76, 165.86, 148.92, 119.27, 42.58, 166.6, 78.55, 381.01, 198.56, 210.49, 162.64, 399.8, 356.24, 262.91, 246.89, 154.65, 384.22, 238.81, 234.92], [318.66, 118.47, 445.62, 168.12, 393.61, 362.11, 254.28, 209.01, 234.39, 121.17, 179.4, 74.01, 356.39, 317.24, 183.98, 417.2, 463.7, 129.69, 489.94, 188.81, 160.16, 126.19, 412.79, 509.84, 267.9, 94.18, 67.08, 163.78, 0.0, 356.38, 389.72, 468.39, 58.26, 312.46, 270.2, 194.26, 184.1, 111.65, 328.73, 362.01, 262.94, 298.56, 451.25, 411.53, 202.05, 136.5, 
288.01, 335.87, 277.77, 390.33], [640.19, 237.93, 230.1, 294.11, 40.0, 149.68, 488.21, 476.07, 254.47, 387.87, 415.65, 391.69, 398.9, 292.31, 255.88, 435.49, 114.0, 274.97, 173.99, 
305.5, 285.51, 305.0, 326.34, 383.36, 275.47, 274.31, 308.76, 196.02, 356.38, 0.0, 292.44, 275.07, 344.08, 57.97, 92.2, 182.27, 326.4, 251.29, 471.26, 37.64, 249.9, 103.41, 369.08, 
330.86, 386.43, 406.51, 111.62, 470.18, 278.65, 49.19], [532.92, 317.35, 122.43, 225.28, 291.21, 145.06, 338.5, 360.22, 155.35, 325.23, 307.53, 367.81, 147.19, 79.06, 428.69, 157.08, 303.16, 260.38, 251.08, 210.24, 439.84, 266.58, 34.48, 122.07, 123.94, 370.41, 391.64, 323.49, 389.72, 292.44, 0.0, 108.69, 339.0, 317.59, 251.32, 351.4, 488.0, 297.6, 257.66, 329.44, 127.01, 202.71, 79.71, 38.47, 257.86, 332.6, 201.32, 251.1, 114.76, 269.42], [639.46, 377.51, 45.54, 314.46, 259.05, 135.79, 446.22, 465.32, 240.53, 421.1, 410.18, 458.19, 253.07, 182.39, 469.88, 250.34, 244.35, 339.89, 171.64, 304.04, 487.71, 352.94, 118.93, 117.02, 218.59, 431.02, 457.56, 369.76, 468.39, 275.07, 108.69, 0.0, 423.56, 317.13, 269.56, 387.6, 535.89, 364.95, 365.64, 312.16, 212.19, 216.6, 136.95, 125.26, 361.45, 430.85, 220.86, 358.71, 211.15, 237.01], [303.49, 115.11, 404.45, 114.24, 378.61, 326.99, 207.02, 167.69, 184.82, 71.03, 127.77, 47.8, 298.41, 264.2, 216.36, 359.22, 444.93, 85.01, 461.57, 133.09, 199.12, 72.45, 360.17, 457.47, 215.6, 117.9, 105.23, 165.86, 58.26, 344.08, 339.0, 423.56, 
0.0, 307.42, 253.31, 204.17, 231.09, 93.41, 272.46, 355.65, 213.68, 272.28, 397.16, 358.52, 144.9, 89.68, 261.23, 279.23, 225.25, 372.15], [608.84, 195.53, 273.64, 275.18, 97.16, 183.96, 469.4, 451.4, 249.6, 358.42, 392.76, 355.18, 406.65, 303.06, 198.65, 449.45, 170.66, 247.42, 231.23, 290.41, 229.01, 278.4, 352.06, 419.05, 277.22, 225.44, 259.83, 148.92, 312.46, 57.97, 317.59, 317.13, 307.42, 0.0, 76.3, 128.46, 268.91, 214.11, 466.48, 49.66, 251.68, 115.32, 396.65, 355.83, 369.48, 377.9, 118.71, 466.69, 282.29, 106.63], [548.0, 152.61, 230.23, 203.91, 125.3, 134.18, 398.58, 384.51, 173.42, 295.68, 324.45, 300.67, 331.12, 228.88, 205.27, 375.54, 193.54, 182.78, 227.5, 217.19, 228.37, 212.9, 285.62, 360.84, 201.1, 
196.12, 229.63, 119.27, 270.2, 92.2, 251.32, 269.56, 253.31, 76.3, 0.0, 122.65, 274.7, 161.64, 390.62, 112.68, 175.64, 53.04, 331.02, 288.77, 297.44, 314.34, 50.77, 390.66, 206.41, 
120.17], [506.43, 90.87, 350.6, 217.06, 222.23, 254.4, 395.49, 365.95, 228.66, 267.57, 314.03, 249.61, 394.68, 307.75, 82.86, 448.43, 296.17, 172.82, 345.18, 239.05, 106.48, 201.43, 383.98, 468.96, 266.05, 101.51, 134.83, 42.58, 194.26, 182.27, 351.4, 387.6, 204.17, 128.46, 122.65, 0.0, 152.09, 120.5, 423.02, 177.08, 245.02, 173.05, 429.38, 385.81, 305.48, 287.68, 167.24, 426.04, 274.48, 226.32], [488.44, 174.4, 500.73, 305.9, 366.03, 404.95, 436.9, 393.02, 346.01, 301.4, 358.83, 258.09, 501.38, 432.15, 70.83, 560.12, 439.52, 255.81, 495.26, 330.05, 48.41, 272.61, 518.35, 608.91, 384.8, 127.46, 126.3, 166.6, 184.1, 326.4, 488.0, 535.89, 231.09, 268.91, 274.7, 152.09, 0.0, 201.52, 499.17, 312.19, 369.01, 324.23, 562.47, 519.17, 371.36, 318.63, 317.65, 504.87, 394.42, 373.43], [395.41, 32.02, 338.64, 106.61, 286.69, 251.88, 275.0, 246.42, 147.09, 148.86, 193.54, 141.21, 300.13, 234.08, 160.1, 358.62, 354.75, 57.07, 378.32, 130.6, 157.01, 82.22, 325.04, 419.65, 185.35, 75.29, 94.15, 78.55, 111.65, 251.29, 297.6, 364.95, 93.41, 214.11, 161.64, 120.5, 201.52, 0.0, 309.66, 262.24, 172.59, 186.96, 367.37, 325.0, 187.27, 168.94, 176.38, 313.82, 195.21, 281.52], [304.72, 341.6, 377.66, 206.0, 488.43, 361.01, 117.55, 165.68, 217.61, 209.91, 154.4, 264.61, 114.95, 198.8, 468.96, 150.0, 526.77, 252.87, 495.8, 184.22, 460.76, 229.17, 250.01, 323.12, 197.37, 374.64, 373.5, 381.01, 328.73, 471.26, 257.66, 365.64, 272.46, 466.48, 390.62, 423.02, 499.17, 309.66, 0.0, 500.59, 222.27, 368.29, 256.46, 244.25, 128.08, 199.83, 359.65, 10.44, 196.21, 470.98], [655.85, 244.75, 267.02, 316.46, 62.43, 187.24, 511.16, 495.76, 283.02, 404.49, 436.25, 403.45, 432.38, 326.34, 243.05, 470.66, 130.08, 292.41, 201.36, 329.82, 274.66, 323.14, 363.46, 420.96, 306.62, 275.02, 309.38, 198.56, 362.01, 37.64, 329.44, 
312.16, 355.65, 49.66, 112.68, 177.08, 312.19, 262.24, 500.59, 0.0, 280.92, 135.79, 406.43, 367.89, 410.11, 423.69, 142.64, 500.01, 310.46, 79.93], [441.99, 195.24, 199.91, 102.31, 
266.18, 147.8, 260.42, 266.24, 28.86, 209.77, 206.7, 246.17, 156.12, 63.51, 315.15, 204.94, 307.94, 133.42, 290.11, 93.06, 321.72, 141.79, 152.63, 248.34, 25.71, 246.95, 266.01, 210.49, 262.94, 249.9, 127.01, 212.19, 213.68, 251.68, 175.64, 245.02, 369.01, 172.59, 222.27, 280.92, 0.0, 146.54, 194.83, 152.43, 162.41, 220.95, 138.52, 220.64, 31.06, 249.02], [554.48, 186.07, 177.86, 200.89, 122.2, 81.61, 391.57, 384.63, 152.64, 303.56, 323.34, 317.77, 296.59, 190.76, 255.89, 336.01, 176.67, 192.78, 191.39, 208.66, 277.0, 219.91, 237.19, 308.89, 172.07, 235.05, 266.83, 162.64, 298.56, 103.41, 202.71, 216.6, 272.28, 115.32, 53.04, 173.05, 324.23, 186.96, 368.29, 135.79, 146.54, 0.0, 282.12, 240.8, 289.52, 320.67, 11.18, 367.01, 175.28, 108.19], [553.09, 389.67, 170.79, 283.32, 364.61, 219.79, 356.88, 388.42, 220.84, 371.95, 343.43, 419.87, 142.58, 134.53, 505.22, 119.62, 367.84, 324.52, 304.53, 264.33, 514.66, 325.38, 45.65, 68.07, 184.17, 441.74, 460.4, 399.8, 451.25, 369.08, 79.71, 136.95, 397.16, 396.65, 331.02, 429.38, 562.47, 367.37, 256.46, 406.43, 194.83, 282.12, 0.0, 43.6, 293.64, 375.69, 280.96, 247.7, 174.18, 342.42], [531.52, 346.72, 149.38, 244.31, 329.04, 183.01, 335.41, 362.4, 179.18, 337.66, 313.64, 383.6, 129.77, 94.37, 461.65, 126.75, 
338.35, 283.67, 281.78, 226.59, 471.29, 286.3, 6.4, 99.3, 143.66, 399.06, 418.4, 356.24, 411.53, 330.86, 38.47, 125.26, 358.52, 355.83, 288.77, 385.81, 519.17, 325.0, 244.25, 367.89, 152.43, 240.8, 43.6, 0.0, 263.55, 342.94, 239.08, 236.58, 133.76, 307.13], [280.44, 219.18, 357.66, 94.37, 411.3, 309.32, 102.07, 104.68, 142.41, 88.53, 50.09, 143.54, 168.65, 179.61, 343.98, 227.48, 462.93, 132.67, 452.1, 81.06, 333.9, 105.08, 267.27, 360.35, 146.54, 247.93, 245.49, 262.91, 202.05, 386.43, 257.86, 361.45, 144.9, 369.48, 297.44, 305.48, 371.36, 187.27, 128.08, 410.11, 162.41, 289.52, 293.64, 263.55, 0.0, 85.48, 279.12, 134.46, 152.12, 397.6], [234.2, 196.82, 420.86, 124.06, 437.14, 358.5, 118.27, 78.75, 194.81, 20.12, 
45.49, 65.8, 254.12, 253.57, 305.95, 312.82, 497.32, 131.59, 500.02, 127.91, 288.56, 101.67, 346.08, 441.28, 212.29, 207.33, 194.04, 246.89, 136.5, 406.51, 332.6, 430.85, 89.68, 377.9, 314.34, 287.68, 318.63, 168.94, 199.83, 423.69, 220.95, 320.67, 375.69, 342.94, 85.48, 0.0, 309.55, 208.22, 220.06, 427.09], [543.41, 176.28, 183.38, 190.03, 132.28, 87.32, 381.18, 373.77, 143.27, 292.41, 312.51, 306.65, 290.24, 185.13, 249.95, 331.07, 187.77, 181.6, 201.95, 198.17, 270.12, 208.75, 235.72, 310.08, 164.17, 225.89, 257.32, 154.65, 288.01, 111.62, 201.32, 220.86, 261.23, 118.71, 50.77, 167.24, 317.65, 176.38, 359.65, 142.64, 138.52, 11.18, 280.96, 239.08, 279.12, 309.55, 0.0, 358.58, 167.84, 118.87], [314.92, 345.68, 371.71, 209.29, 486.57, 357.46, 127.95, 175.79, 217.33, 217.75, 162.89, 272.66, 106.93, 194.37, 473.47, 140.01, 523.64, 256.88, 491.13, 187.0, 465.93, 233.98, 242.42, 314.09, 195.45, 
379.82, 379.49, 384.22, 335.87, 470.18, 251.1, 358.71, 279.23, 466.69, 390.66, 426.04, 504.87, 313.82, 10.44, 500.01, 220.64, 367.01, 247.7, 236.58, 134.46, 208.22, 358.58, 0.0, 193.78, 468.82], [432.39, 220.06, 205.55, 111.07, 293.06, 166.9, 245.1, 256.78, 48.41, 211.4, 199.68, 253.05, 125.06, 39.66, 342.71, 174.45, 331.01, 150.34, 306.73, 95.52, 347.7, 152.85, 135.09, 232.31, 10.0, 270.35, 287.01, 238.81, 277.77, 278.65, 114.76, 211.15, 225.25, 282.29, 206.41, 274.48, 394.42, 195.21, 196.21, 310.46, 31.06, 175.28, 174.18, 133.76, 152.12, 220.06, 167.84, 193.78, 0.0, 275.07], [661.25, 272.42, 191.48, 308.85, 22.36, 124.49, 499.61, 492.56, 259.01, 409.4, 431.34, 419.04, 388.96, 282.12, 302.68, 419.7, 73.38, 297.12, 124.81, 316.81, 331.3, 325.65, 302.04, 349.49, 273.62, 313.25, 347.38, 234.92, 390.33, 49.19, 269.42, 237.01, 372.15, 106.63, 120.17, 226.32, 373.43, 281.52, 470.98, 79.93, 249.02, 108.19, 342.42, 307.13, 397.6, 427.09, 118.87, 468.82, 275.07, 0.0],
                      
                         ])     
                             
                            


    #"""To form the the Demand"""
    # if count ==1:
    data['demands'] =[0, 25, 38, 21, 34, 21, 18, 29, 29, 21, 39, 12, 40, 16, 17, 32, 34, 28, 19, 37, 13, 23, 33, 45, 42, 14, 36, 48, 24, 
                      37, 49, 22, 47, 48, 16, 37, 43, 38, 48, 21, 14, 28, 14, 48, 17, 10, 31, 42, 43, 20]

    data['num_vehicles'] = 4
    data['vehicle_capacities'] = [200,200,200,200]


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