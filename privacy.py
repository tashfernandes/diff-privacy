##################################
# Differential privacy algorithms
##################################

import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Let's test out that these are all the same thing.

# 1. Laplace distribution: b * sign(u) * log(1-2*abs(u)) for u in (-0.5,0.5), b is parameter.
def laplace_dist(b):
    u = random.random() - 0.5
    sign = 1.0
    if u < 0:
        sign = -1.0
    return b * sign * math.log(1.0 - 2.0 * abs(u))

# 2. If U and V are random variables following an exponential distribution then Y = U - V 
# follows the laplace distribution.
def laplace_dist2(b):
    U = np.random.exponential(scale=b)
    V = np.random.exponential(scale=b)
    return U - V

# 3. Use the laplace function in numpy
def laplace_dist3(b):
    return np.random.laplace(scale=b)
    

#####################


### main

x = []
for i in range(10000):
    x.append(laplace_dist3(1.0))

n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
plt.grid(True)
plt.show()

