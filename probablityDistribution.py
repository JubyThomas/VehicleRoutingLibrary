import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
    
# Plot between -30 and 30 with
# 0.1 steps.
x_axis = np.linspace(1,50,200)
    
# Calculating mean and standard 
# deviation
mean = statistics.mean(x_axis)
print(mean)
sd = statistics.stdev(x_axis)
print(sd)
    
plt.plot(x_axis, norm.pdf(x_axis, mean, sd))
plt.show()

print(sum([10,12,13]))