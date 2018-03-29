import dlib
from math import sin,cos,pi,exp,sqrt

# This is a standard test function for these kinds of optimization problems.
# It has a bunch of local minima, with the global minimum resulting in
# holder_table()==-19.2085025679.
def holder_table(x0,x1):
    return -abs(sin(x0)*cos(x1)*exp(abs(1-sqrt(x0*x0+x1*x1)/pi)))

# Find the optimal inputs to holder_table().  The print statements that follow
# show that find_min_global() finds the optimal settings to high precision.
x,y = dlib.find_min_global(holder_table,
                           [-10,-10],  # Lower bound constraints on x0 and x1 respectively
                           [10,10],    # Upper bound constraints on x0 and x1 respectively
                           80)         # The number of times find_min_global() will call holder_table()

print("optimal inputs: {}".format(x));
print("optimal output: {}".format(y));