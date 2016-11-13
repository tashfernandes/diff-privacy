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
    

# Use Gamma distribution when using l2 norm as the metric for determining
# distance between 2 datasets.

# NOTES:
# To add noise to the logistic regression function we need to use a noise vector
# and add it to the calculated weight vector.
# To do this we need to select the norm and direction for the vector.
# The vector direction is selected from a gaussian distribution (each dimension
# picked and then vector normalised). See http://mathworld.wolfram.com/HyperspherePointPicking.html
# The norm is selected from the Gamma distribution.

def generate_unit_vector(dimensions):
    '''Selected a random (normalised) vector using Gaussian distribution.'''
    v = np.random.normal(size=dimensions)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

#
# Inputs:
# lda - the 'lambda' value calculated from regularisation
# training_set_size - the number of training examples
# epsilon - the value for the privacy budget
#
def generate_noise_vector(dimensions, lda, epsilon, training_set_size):
    '''Generate a noise vector for logistic regression according to Gamma distribution.'''
    v_dir = generate_unit_vector(dimensions)
    scale = 2 / (training_set_size * epsilon * lda)
    v_length = np.random.gamma(dimensions, scale=scale)
    return v_dir * v_length


#####################

def test_unit_vector():
    v = generate_unit_vector(10)
    print "Should be on unit circle: %f" % sum(v * v)

### main

test_unit_vector()

#x = []
#for i in range(10000):
#    x.append(laplace_dist3(1.0))
#
#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
#plt.grid(True)
#plt.show()

