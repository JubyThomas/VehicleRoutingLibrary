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
[0.0, 403.41, 435.04, 194.49, 207.14, 589.9, 347.09, 512.96, 336.54, 238.47, 558.62, 419.9, 467.75, 396.54, 330.55, 448.98, 543.8, 506.72, 456.44, 308.36, 155.57, 560.06, 495.55, 404.77, 412.69, 410.21, 430.75, 352.14, 298.74, 528.82, 429.78, 430.69, 394.09, 605.08, 370.24, 144.55, 479.32, 464.01, 245.17, 397.82, 395.05, 413.36, 456.07, 285.04, 253.14, 227.43, 202.2, 231.82, 448.16, 478.8], [403.41, 0.0, 137.71, 224.41, 201.6, 247.7, 192.37, 109.77, 82.49, 185.31, 213.23, 110.94, 106.3, 287.98, 146.32, 53.71, 198.28, 282.58, 434.63, 111.83, 259.09, 218.51, 92.7, 165.25, 108.41, 413.27, 30.02, 194.13, 110.22, 253.81, 257.87, 286.76, 137.54, 265.94, 107.0, 310.42, 120.07, 153.15, 319.13, 193.2, 78.85, 302.85, 181.62, 137.61, 180.07, 215.23, 215.34, 333.19, 215.26, 121.08], [435.04, 137.71, 0.0, 298.1, 233.01, 361.73, 111.2, 163.77, 122.38, 272.47, 329.76, 27.59, 232.69, 420.86, 105.02, 171.03, 316.67, 419.51, 565.86, 138.65, 321.89, 336.15, 150.85, 48.55, 246.11, 540.58, 125.87, 108.9, 203.12, 386.76, 394.61, 422.76, 274.24, 379.53, 240.47, 307.42, 
66.73, 29.07, 422.91, 329.25, 65.3, 436.84, 318.43, 155.26, 275.26, 209.55, 233.34, 433.24, 80.05, 65.0], [194.49, 224.41, 298.1, 0.0, 91.05, 395.67, 251.59, 332.96, 178.36, 44.2, 364.17, 275.39, 274.87, 236.0, 210.19, 263.14, 349.32, 325.73, 342.7, 159.62, 38.95, 365.69, 316.54, 286.0, 218.34, 302.91, 
253.97, 256.84, 114.24, 339.76, 254.03, 263.39, 200.12, 411.1, 175.93, 171.0, 323.19, 325.96, 142.34, 210.3, 241.51, 255.02, 263.8, 147.73, 58.8, 149.03, 110.45, 146.1, 341.28, 323.26], [207.14, 201.6, 233.01, 91.05, 0.0, 419.37, 166.39, 309.42, 129.62, 100.44, 385.36, 215.13, 282.8, 312.25, 133.0, 251.46, 369.76, 382.15, 430.11, 101.24, 97.06, 388.49, 291.73, 210.64, 239.71, 391.88, 226.95, 171.76, 111.89, 383.52, 319.01, 334.35, 235.29, 436.5, 200.38, 115.1, 272.68, 262.03, 233.31, 263.83, 188.22, 331.48, 302.64, 78.77, 116.4, 58.31, 23.02, 236.18, 262.25, 272.25], [589.9, 247.7, 361.73, 395.67, 419.37, 0.0, 439.12, 206.9, 326.64, 352.21, 34.79, 341.02, 142.72, 272.81, 394.0, 197.23, 50.33, 161.79, 388.09, 352.6, 434.44, 30.89, 214.57, 402.01, 180.57, 394.37, 242.67, 440.45, 307.51, 102.12, 216.52, 241.87, 195.87, 18.25, 222.2, 534.47, 308.13, 362.75, 406.99, 205.53, 320.14, 272.52, 138.97, 375.94, 336.88, 450.52, 438.95, 426.25, 440.91, 310.1], [347.09, 192.37, 111.2, 251.59, 166.39, 439.12, 0.0, 260.76, 124.66, 241.83, 404.98, 113.72, 298.67, 431.84, 52.55, 242.0, 390.27, 462.17, 567.65, 114.06, 263.41, 410.47, 244.9, 65.86, 289.07, 535.05, 197.52, 5.39, 195.23, 442.0, 419.51, 443.2, 306.1, 457.36, 267.17, 208.89, 175.73, 135.35, 391.41, 354.57, 120.6, 450.0, 364.48, 109.88, 251.38, 122.0, 156.85, 397.2, 101.08, 174.23], [512.96, 109.77, 163.77, 332.96, 309.42, 206.9, 260.76, 0.0, 182.88, 292.32, 178.84, 149.27, 112.33, 350.91, 228.08, 75.11, 168.67, 304.06, 496.03, 213.34, 368.41, 185.88, 17.8, 209.62, 163.17, 483.01, 82.49, 260.45, 218.87, 257.0, 307.51, 338.75, 200.5, 223.66, 187.3, 414.21, 103.04, 159.0, 415.24, 252.24, 143.39, 361.8, 211.01, 239.43, 285.07, 315.69, 321.31, 431.01, 239.3, 105.04], [336.54, 82.49, 122.38, 178.36, 129.62, 326.64, 124.66, 182.88, 0.0, 150.22, 291.89, 97.8, 183.93, 312.88, 72.84, 136.19, 276.54, 337.55, 453.72, 30.48, 206.04, 296.64, 165.08, 123.75, 165.44, 425.0, 101.82, 127.88, 82.2, 319.23, 296.28, 321.18, 181.46, 344.78, 142.64, 231.37, 146.41, 148.96, 300.61, 230.85, 63.2, 330.35, 240.52, 56.64, 152.97, 133.79, 139.09, 310.86, 181.07, 146.28], [238.47, 185.31, 272.47, 44.2, 100.44, 352.21, 241.83, 292.32, 150.22, 0.0, 320.41, 247.94, 230.76, 211.91, 195.06, 221.05, 305.48, 289.83, 332.36, 137.56, 83.1, 322.08, 276.33, 266.85, 174.23, 296.22, 215.23, 246.81, 76.0, 299.67, 221.9, 235.14, 157.2, 367.85, 131.8, 201.07, 290.5, 299.15, 152.12, 172.35, 211.99, 231.14, 222.09, 132.55, 17.46, 156.87, 123.17, 161.06, 323.74, 290.76], [558.62, 213.23, 329.76, 364.17, 385.36, 34.79, 404.98, 178.84, 291.89, 320.41, 0.0, 308.37, 107.97, 257.03, 359.49, 163.37, 15.65, 157.8, 381.08, 317.81, 403.05, 7.21, 184.89, 369.09, 147.43, 383.33, 209.12, 406.39, 273.59, 
98.62, 201.3, 229.02, 165.42, 52.89, 189.5, 500.44, 278.19, 332.03, 382.9, 181.24, 286.45, 259.03, 114.02, 341.18, 305.47, 415.89, 404.74, 401.98, 409.32, 280.13], [419.9, 110.94, 27.59, 275.39, 215.13, 341.02, 113.72, 149.27, 97.8, 247.94, 308.37, 0.0, 208.69, 393.28, 94.41, 147.31, 294.82, 393.29, 538.31, 117.21, 300.88, 314.57, 134.63, 61.4, 219.32, 513.21, 101.41, 112.68, 177.13, 361.87, 367.24, 395.29, 246.92, 359.0, 212.93, 297.2, 64.4, 51.26, 397.47, 301.81, 38.01, 409.26, 292.23, 136.44, 249.95, 197.59, 217.7, 408.24, 104.75, 63.25], [467.75, 106.3, 232.69, 274.87, 282.8, 142.72, 298.67, 112.33, 183.93, 230.76, 107.97, 208.69, 0.0, 241.02, 252.04, 61.68, 92.65, 195.06, 384.66, 210.09, 313.52, 112.81, 108.25, 267.57, 63.15, 373.79, 107.28, 300.37, 172.74, 154.95, 195.53, 226.82, 98.02, 160.85, 101.64, 397.25, 191.76, 240.31, 325.71, 143.59, 182.46, 250.8, 98.86, 233.86, 218.32, 309.48, 300.94, 343.28, 312.67, 193.49], [396.54, 287.98, 420.86, 236.0, 312.25, 272.81, 431.84, 350.91, 312.88, 211.91, 257.03, 393.28, 241.02, 0.0, 379.79, 282.18, 
249.16, 128.14, 147.05, 317.8, 263.6, 253.66, 343.02, 435.21, 189.4, 132.85, 310.8, 436.01, 236.65, 175.91, 56.29, 37.48, 153.39, 279.96, 181.22, 407.0, 406.62, 439.97, 161.01, 98.84, 356.05, 19.24, 151.2, 325.33, 195.92, 367.45, 335.06, 179.45, 491.48, 407.84], [330.55, 146.32, 105.02, 210.19, 133.0, 394.0, 52.55, 228.08, 72.84, 195.06, 359.49, 94.41, 252.04, 379.79, 0.0, 198.37, 344.44, 410.28, 516.81, 62.13, 227.4, 364.66, 211.07, 77.67, 238.11, 485.17, 156.39, 56.36, 143.42, 391.68, 366.95, 390.71, 253.95, 412.24, 214.9, 203.16, 158.47, 133.75, 346.54, 302.03, 84.81, 397.85, 313.31, 62.97, 203.09, 104.55, 130.0, 353.73, 131.18, 157.44], [448.98, 53.71, 171.03, 263.14, 251.46, 197.23, 242.0, 75.11, 136.19, 221.05, 163.37, 147.31, 61.68, 282.18, 198.37, 0.0, 149.05, 253.51, 428.78, 165.3, 299.93, 169.14, 62.82, 206.87, 92.78, 412.83, 46.17, 243.25, 150.72, 216.34, 243.32, 274.01, 129.25, 215.46, 112.45, 362.37, 132.41, 179.01, 340.15, 183.5, 123.1, 294.59, 153.84, 190.88, 212.46, 268.43, 266.59, 356.0, 251.03, 134.03], [543.8, 198.28, 316.67, 349.32, 369.76, 50.33, 390.27, 168.67, 276.54, 305.48, 15.65, 294.82, 92.65, 249.16, 344.44, 149.05, 0.0, 156.26, 376.75, 302.31, 388.23, 20.22, 173.62, 355.32, 132.14, 377.19, 195.0, 391.75, 258.01, 98.79, 194.02, 222.65, 151.21, 68.31, 174.32, 484.83, 266.48, 319.76, 371.02, 169.81, 272.15, 252.2, 102.88, 325.61, 290.69, 400.25, 389.1, 390.0, 396.42, 268.4], [506.72, 282.58, 419.51, 325.73, 382.15, 161.79, 462.17, 304.06, 337.55, 289.83, 157.8, 393.29, 
195.06, 128.14, 410.28, 253.51, 156.26, 0.0, 226.37, 353.28, 360.25, 152.2, 302.54, 446.86, 175.23, 235.33, 294.98, 465.24, 281.04, 60.02, 78.75, 91.42, 156.7, 164.01, 196.01, 490.59, 385.81, 431.27, 284.96, 118.38, 360.81, 121.2, 101.08, 369.14, 272.37, 429.42, 404.86, 304.07, 497.82, 387.47], [456.44, 434.63, 565.86, 342.7, 430.11, 388.09, 567.65, 496.03, 453.72, 332.36, 381.08, 538.31, 384.66, 147.05, 516.81, 428.78, 376.75, 226.37, 0.0, 454.8, 358.73, 376.19, 488.95, 577.25, 336.02, 46.53, 457.82, 572.17, 373.89, 286.01, 189.62, 158.4, 300.37, 390.16, 327.67, 506.42, 553.6, 585.89, 216.45, 245.28, 500.7, 134.33, 289.34, 458.46, 319.39, 488.07, 451.56, 226.18, 634.2, 554.8], [308.36, 111.83, 138.65, 159.62, 101.24, 352.6, 114.06, 213.34, 30.48, 137.56, 317.81, 117.21, 210.09, 317.8, 62.13, 165.3, 302.31, 353.28, 454.8, 0.0, 183.7, 322.26, 195.53, 129.32, 185.33, 423.62, 132.25, 118.21, 81.3, 339.18, 306.43, 329.46, 196.6, 370.62, 157.28, 200.89, 171.82, 166.88, 289.66, 242.17, 87.3, 335.94, 259.03, 26.17, 143.6, 103.59, 109.18, 298.32, 186.39, 171.47], [155.57, 259.09, 321.89, 38.95, 97.06, 434.44, 263.41, 368.41, 206.04, 83.1, 403.05, 300.88, 313.52, 263.6, 227.4, 299.93, 388.23, 360.25, 358.73, 183.7, 0.0, 404.53, 351.67, 304.8, 257.29, 316.55, 288.2, 268.78, 149.7, 376.78, 286.59, 293.39, 238.76, 449.77, 214.87, 147.87, 352.32, 350.41, 147.58, 246.53, 269.0, 282.2, 301.81, 167.49, 97.59, 149.94, 110.42, 146.17, 358.06, 352.24], [560.06, 218.51, 336.15, 365.69, 388.49, 30.89, 410.47, 185.88, 296.64, 322.08, 7.21, 314.57, 112.81, 253.66, 364.66, 169.14, 20.22, 152.2, 376.19, 322.26, 404.53, 0.0, 191.76, 375.19, 149.84, 379.13, 215.02, 411.95, 276.63, 92.7, 197.73, 225.02, 166.35, 48.47, 191.64, 503.59, 284.99, 338.69, 381.69, 179.92, 292.25, 255.2, 112.73, 345.42, 306.94, 419.79, 408.06, 400.84, 415.79, 286.92], [495.55, 92.7, 150.85, 316.54, 291.73, 214.57, 244.9, 17.8, 165.08, 276.33, 184.89, 134.63, 108.25, 343.02, 211.07, 62.82, 173.62, 302.54, 488.95, 195.53, 351.67, 191.76, 0.0, 195.56, 154.03, 474.47, 64.88, 244.77, 202.34, 257.95, 301.48, 332.57, 191.08, 231.81, 175.14, 396.41, 93.56, 148.57, 401.78, 244.18, 126.57, 354.62, 206.96, 221.63, 269.55, 297.93, 303.53, 417.23, 228.01, 95.52], [404.77, 165.25, 48.55, 286.0, 210.64, 402.01, 65.86, 209.62, 123.75, 266.85, 369.09, 61.4, 267.57, 435.21, 77.67, 206.87, 355.32, 446.86, 577.25, 129.32, 304.8, 375.19, 195.56, 0.0, 271.68, 548.73, 160.7, 62.68, 205.37, 418.87, 414.54, 441.08, 295.62, 420.04, 258.99, 270.96, 115.25, 69.81, 418.97, 348.68, 86.4, 452.23, 346.3, 138.35, 272.66, 177.34, 206.62, 427.38, 57.49, 113.53], [412.69, 108.41, 246.11, 218.34, 239.71, 180.57, 289.07, 163.17, 165.44, 174.23, 147.43, 219.32, 63.15, 189.4, 238.11, 92.78, 132.14, 175.23, 336.02, 185.33, 257.29, 149.84, 154.03, 271.68, 0.0, 320.45, 125.32, 291.81, 127.95, 154.0, 152.07, 182.22, 37.36, 197.27, 
42.45, 354.71, 220.64, 260.42, 262.62, 90.74, 185.81, 201.91, 75.5, 205.2, 160.24, 275.28, 260.03, 280.14, 323.34, 222.04], [410.21, 413.27, 540.58, 302.91, 391.88, 394.37, 535.05, 483.01, 425.0, 296.22, 383.33, 513.21, 373.79, 132.85, 485.17, 412.83, 377.19, 235.33, 46.53, 423.62, 316.55, 379.13, 474.47, 548.73, 320.45, 0.0, 438.18, 539.74, 343.73, 292.83, 183.99, 154.55, 283.58, 398.54, 306.66, 463.73, 533.19, 562.03, 172.0, 230.83, 475.28, 125.6, 283.01, 425.29, 284.46, 450.12, 412.78, 180.71, 606.06, 534.27], [430.75, 30.02, 125.87, 253.97, 226.95, 242.67, 197.52, 82.49, 101.82, 215.23, 209.12, 101.41, 107.28, 310.8, 156.39, 46.17, 195.0, 294.98, 457.82, 132.25, 288.2, 215.02, 64.88, 160.7, 125.32, 438.18, 0.0, 198.48, 139.9, 260.89, 277.2, 306.97, 158.14, 260.85, 131.61, 332.93, 95.9, 136.34, 348.53, 214.01, 77.52, 324.85, 194.14, 158.42, 210.09, 235.55, 239.2, 362.8, 205.65, 97.19], [352.14, 194.13, 108.9, 256.84, 171.76, 440.45, 5.39, 260.45, 127.88, 246.81, 406.39, 112.68, 300.37, 436.01, 56.36, 243.25, 391.75, 465.24, 572.17, 118.21, 268.78, 411.95, 244.77, 62.68, 291.81, 539.74, 198.48, 0.0, 199.46, 444.49, 423.2, 447.07, 309.34, 458.68, 270.49, 213.59, 174.0, 132.41, 396.55, 358.15, 121.28, 454.12, 367.27, 114.69, 256.22, 127.28, 162.23, 402.4, 96.02, 172.47], [298.74, 110.22, 203.12, 114.24, 111.89, 307.51, 195.23, 218.87, 82.2, 76.0, 273.59, 177.13, 172.74, 236.65, 143.42, 150.72, 258.01, 281.04, 373.89, 81.3, 149.7, 276.63, 202.34, 205.37, 127.95, 343.73, 139.9, 199.46, 0.0, 275.35, 227.98, 249.56, 127.28, 324.61, 89.99, 226.99, 215.38, 228.25, 220.63, 165.6, 139.85, 254.91, 193.96, 90.25, 73.98, 149.93, 132.1, 232.18, 262.84, 215.74], [528.82, 253.81, 386.76, 339.76, 383.52, 102.12, 442.0, 257.0, 319.23, 299.67, 98.62, 361.87, 154.95, 175.91, 391.68, 216.34, 98.79, 60.02, 286.01, 339.18, 376.78, 92.7, 257.95, 418.87, 154.0, 292.83, 260.89, 444.49, 275.35, 0.0, 120.42, 142.37, 148.43, 105.8, 185.61, 496.23, 346.16, 395.24, 322.62, 131.02, 332.57, 173.17, 81.39, 358.26, 282.64, 425.22, 405.38, 342.02, 466.46, 347.94], [429.78, 257.87, 394.61, 254.03, 319.01, 216.52, 419.51, 307.51, 296.28, 221.9, 201.3, 367.24, 195.53, 56.29, 366.95, 243.32, 194.02, 78.75, 189.62, 306.43, 286.59, 197.73, 301.48, 414.54, 152.07, 183.99, 277.2, 423.2, 227.98, 120.42, 0.0, 31.3, 120.42, 223.76, 155.55, 422.59, 372.7, 411.0, 206.55, 65.86, 331.44, 58.42, 99.93, 318.22, 204.63, 370.38, 342.01, 225.75, 469.08, 374.09], [430.69, 286.76, 422.76, 263.39, 334.35, 241.87, 443.2, 338.75, 321.18, 235.14, 229.02, 395.29, 226.82, 37.48, 390.71, 274.01, 222.65, 91.42, 158.4, 329.46, 293.39, 225.02, 332.57, 441.08, 182.22, 154.55, 306.97, 447.07, 249.56, 142.37, 31.3, 0.0, 149.28, 247.7, 182.63, 433.84, 402.67, 439.89, 198.04, 93.61, 359.01, 30.81, 130.97, 339.63, 218.33, 387.67, 357.35, 216.64, 496.23, 404.03], [394.09, 137.54, 274.24, 200.12, 235.29, 195.87, 306.1, 200.5, 181.46, 157.2, 165.42, 246.92, 98.02, 153.39, 253.95, 129.25, 151.21, 156.7, 300.37, 196.6, 238.76, 166.35, 191.08, 295.62, 37.36, 283.58, 158.14, 309.34, 127.28, 148.43, 120.42, 149.28, 0.0, 211.02, 39.32, 348.74, 254.03, 290.69, 228.88, 55.9, 211.47, 166.82, 67.36, 213.08, 141.35, 277.2, 257.0, 246.91, 349.34, 255.33], [605.08, 265.94, 379.53, 411.1, 436.5, 18.25, 457.36, 223.66, 344.78, 367.85, 52.89, 359.0, 160.85, 279.96, 412.24, 215.46, 68.31, 164.01, 390.16, 370.62, 449.77, 48.47, 231.81, 420.04, 197.27, 398.54, 260.85, 458.68, 324.61, 105.8, 223.76, 247.7, 211.02, 0.0, 238.56, 551.6, 325.35, 380.17, 418.14, 217.45, 338.33, 278.49, 151.82, 393.86, 352.3, 468.24, 456.25, 437.48, 458.58, 327.32], [370.24, 107.0, 240.47, 175.93, 200.38, 222.2, 267.17, 187.3, 142.64, 131.8, 189.5, 212.93, 101.64, 181.22, 214.9, 112.45, 174.32, 196.01, 327.67, 157.28, 214.87, 191.64, 175.14, 258.99, 42.45, 306.66, 131.61, 270.49, 89.99, 185.61, 155.55, 182.63, 39.32, 238.56, 0.0, 314.93, 226.6, 258.82, 229.14, 89.69, 176.4, 196.6, 104.24, 174.0, 118.0, 239.65, 221.45, 245.74, 313.68, 227.71], [144.55, 310.42, 307.42, 171.0, 115.1, 534.47, 208.89, 414.21, 231.37, 201.07, 500.44, 297.2, 397.25, 407.0, 203.16, 362.37, 484.83, 490.59, 506.42, 200.89, 147.87, 503.59, 396.41, 270.96, 354.71, 463.73, 332.93, 213.59, 226.99, 496.23, 422.59, 433.84, 348.74, 551.6, 314.93, 0.0, 360.46, 335.52, 292.84, 372.54, 279.89, 426.01, 415.99, 174.77, 218.51, 100.34, 96.4, 288.23, 308.93, 359.57], [479.32, 120.07, 66.73, 323.19, 272.68, 308.13, 175.73, 103.04, 146.41, 290.5, 278.19, 64.4, 191.76, 406.62, 158.47, 132.41, 266.48, 385.81, 553.6, 171.82, 352.32, 284.99, 93.56, 115.25, 220.64, 533.19, 95.9, 174.0, 215.38, 346.16, 372.7, 402.67, 254.03, 325.35, 226.6, 360.46, 0.0, 56.14, 434.07, 309.89, 84.53, 420.75, 286.2, 194.28, 289.28, 260.38, 277.69, 446.76, 136.82, 2.0], [464.01, 153.15, 29.07, 325.96, 262.03, 362.75, 135.35, 159.0, 148.96, 299.15, 332.03, 51.26, 240.31, 439.97, 133.75, 179.01, 319.76, 431.27, 585.89, 166.88, 350.41, 338.69, 148.57, 69.81, 260.42, 562.03, 136.34, 132.41, 228.25, 395.24, 411.0, 439.89, 290.69, 380.17, 258.82, 335.52, 56.14, 0.0, 448.69, 346.3, 88.55, 455.38, 330.48, 184.13, 301.2, 238.2, 262.39, 459.5, 80.75, 54.15], [245.17, 319.13, 422.91, 142.34, 233.31, 406.99, 391.41, 415.24, 300.61, 152.12, 382.9, 397.47, 325.71, 161.01, 346.54, 340.15, 371.02, 284.96, 216.45, 289.66, 147.58, 381.69, 401.78, 418.97, 262.62, 172.0, 348.53, 396.55, 220.63, 322.62, 206.55, 198.04, 228.88, 418.14, 229.14, 292.84, 434.07, 448.69, 0.0, 201.78, 360.44, 175.11, 268.97, 283.65, 147.65, 290.95, 251.81, 19.42, 475.83, 434.61], [397.82, 193.2, 329.25, 210.3, 263.83, 205.53, 354.57, 252.24, 230.85, 172.35, 181.24, 301.81, 143.59, 98.84, 302.03, 183.5, 169.81, 118.38, 245.28, 242.17, 246.53, 179.92, 244.18, 348.68, 90.74, 230.83, 214.01, 358.15, 165.6, 131.02, 65.86, 93.61, 55.9, 217.45, 89.69, 372.54, 309.89, 346.3, 201.78, 0.0, 265.76, 111.25, 67.23, 255.36, 154.95, 312.0, 286.57, 220.95, 403.29, 311.19], [395.05, 78.85, 65.3, 241.51, 188.22, 320.14, 120.6, 143.39, 63.2, 211.99, 286.45, 38.01, 182.46, 356.05, 84.81, 123.1, 272.15, 360.81, 500.7, 87.3, 269.0, 292.25, 126.57, 86.4, 185.81, 475.28, 77.52, 121.28, 139.85, 332.57, 331.44, 359.01, 211.47, 338.33, 176.4, 279.89, 84.53, 88.55, 360.44, 265.76, 0.0, 372.25, 260.07, 110.06, 213.17, 179.57, 193.94, 371.6, 137.93, 84.17], [413.36, 302.85, 436.84, 255.02, 331.48, 272.52, 450.0, 361.8, 330.35, 231.14, 259.03, 409.26, 250.8, 19.24, 397.85, 294.59, 252.2, 121.2, 134.33, 335.94, 282.2, 255.2, 354.62, 452.23, 201.91, 125.6, 324.85, 454.12, 
254.91, 173.17, 58.42, 30.81, 166.82, 278.49, 196.6, 426.01, 420.75, 455.38, 175.11, 111.25, 372.25, 0.0, 157.65, 343.9, 215.14, 386.62, 354.29, 193.04, 508.26, 422.02], [456.07, 181.62, 318.43, 263.8, 302.64, 138.97, 364.48, 211.01, 240.52, 222.09, 114.02, 292.23, 98.86, 151.2, 313.31, 153.84, 102.88, 
101.08, 289.34, 259.03, 301.81, 112.73, 206.96, 346.3, 75.5, 283.01, 194.14, 367.27, 193.96, 81.39, 99.93, 130.97, 67.36, 151.82, 104.24, 415.99, 286.2, 330.48, 268.97, 67.23, 260.07, 157.65, 0.0, 277.41, 205.55, 343.82, 324.32, 288.11, 396.81, 287.8], [285.04, 137.61, 155.26, 147.73, 78.77, 375.94, 109.88, 239.43, 56.64, 132.55, 341.18, 136.44, 233.86, 325.33, 62.97, 190.88, 325.61, 369.14, 458.46, 26.17, 167.49, 345.42, 221.63, 138.35, 205.2, 425.29, 158.42, 114.69, 90.25, 358.26, 318.22, 339.63, 213.08, 393.86, 174.0, 174.77, 194.28, 184.13, 283.65, 255.36, 110.06, 343.9, 277.41, 0.0, 141.53, 77.62, 84.17, 290.76, 193.64, 193.79], [253.14, 180.07, 275.26, 58.8, 116.4, 336.88, 251.38, 285.07, 152.97, 17.46, 305.47, 249.95, 218.32, 195.92, 203.09, 212.46, 290.69, 272.37, 319.39, 143.6, 97.59, 306.94, 269.55, 272.66, 160.24, 284.46, 210.09, 256.22, 73.98, 282.64, 204.63, 218.33, 141.35, 352.3, 118.0, 218.51, 289.28, 301.2, 147.65, 154.95, 213.17, 215.14, 205.55, 141.53, 0.0, 171.84, 139.3, 158.43, 329.91, 289.66], [227.43, 215.23, 209.55, 149.03, 58.31, 450.52, 122.0, 315.69, 133.79, 156.87, 415.89, 197.59, 309.48, 367.45, 104.55, 268.43, 400.25, 429.42, 488.07, 103.59, 149.94, 419.79, 297.93, 177.34, 275.28, 450.12, 235.55, 127.28, 149.93, 425.22, 370.38, 387.67, 277.2, 468.24, 239.65, 100.34, 260.38, 238.2, 290.95, 312.0, 179.57, 386.62, 343.82, 77.62, 171.84, 0.0, 39.62, 293.02, 222.52, 259.54], [202.2, 215.34, 233.34, 110.45, 23.02, 438.95, 156.85, 321.31, 139.09, 123.17, 404.74, 217.7, 300.94, 335.06, 130.0, 266.59, 389.1, 404.86, 451.56, 109.18, 110.42, 408.06, 303.53, 206.62, 260.03, 412.78, 239.2, 162.23, 132.1, 405.38, 342.01, 357.35, 
257.0, 456.25, 221.45, 96.4, 277.69, 262.39, 251.81, 286.57, 193.94, 354.29, 324.32, 84.17, 139.3, 39.62, 0.0, 253.54, 255.56, 277.1], [231.82, 333.19, 
433.24, 146.1, 236.18, 426.25, 397.2, 431.01, 310.86, 161.06, 401.98, 408.24, 343.28, 179.45, 353.73, 356.0, 390.0, 304.07, 226.18, 298.32, 146.17, 400.84, 417.23, 427.38, 280.14, 180.71, 362.8, 402.4, 232.18, 342.02, 225.75, 216.64, 246.91, 437.48, 245.74, 288.23, 446.76, 459.5, 19.42, 220.95, 371.6, 193.04, 288.11, 290.76, 158.43, 293.02, 253.54, 0.0, 483.86, 447.24], [448.16, 215.26, 80.05, 341.28, 262.25, 440.91, 101.08, 239.3, 181.07, 323.74, 409.32, 104.75, 312.67, 491.48, 131.18, 251.03, 396.42, 497.82, 634.2, 186.39, 358.06, 415.79, 228.01, 57.49, 323.34, 606.06, 205.65, 96.02, 262.84, 466.46, 469.08, 496.23, 349.34, 458.58, 313.68, 308.93, 136.82, 80.75, 475.83, 403.29, 137.93, 508.26, 396.81, 193.64, 329.91, 222.52, 255.56, 483.86, 0.0, 134.84], [478.8, 121.08, 65.0, 323.26, 272.25, 310.1, 174.23, 105.04, 146.28, 290.76, 280.13, 63.25, 193.49, 407.84, 157.44, 134.03, 268.4, 387.47, 554.8, 
171.47, 352.24, 286.92, 95.52, 113.53, 222.04, 534.27, 97.19, 172.47, 215.74, 347.94, 374.09, 404.03, 255.33, 327.32, 227.71, 359.57, 2.0, 
54.15, 434.61, 311.19, 84.17, 422.02, 287.8, 193.79, 289.66, 259.54, 277.1, 447.24, 134.84, 0.0]

   
                         ])     
                             
                        
    #"""To form the the Demand"""
    # if count ==1:
    data['demands'] =[0, 48, 15, 40, 38, 35, 13, 43, 28, 32,
                      31, 30, 29, 26, 15, 25, 22, 25, 24, 22, 
                      20, 18, 18, 17, 16, 16, 15, 13,40,17,28,39,
                      13, 23, 12, 28, 26, 32, 40, 18, 19, 29, 31,14, 29,
                      23, 32, 18, 21, 30]
    
    global total_demand_per_day
    total_demand_per_day=sum(data['demands'])
    global number_of_nodes
    number_of_nodes=len(data['demands'])

    data['num_vehicles'] = 4
    global number_of_routes_created
    number_of_routes_created=data['num_vehicles']
    
    data['vehicle_capacities'] = [200,200,200,200]
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