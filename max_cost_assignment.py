import dlib

# Let's imagine you need to assign N people to N jobs.  Additionally, each
# person will make your company a certain amount of money at each job, but each
# person has different skills so they are better at some jobs and worse at
# others.  You would like to find the best way to assign people to these jobs.
# In particular, you would like to maximize the amount of money the group makes
# as a whole.  This is an example of an assignment problem and is what is solved
# by the dlib.max_cost_assignment() routine.

# So in this example, let's imagine we have 3 people and 3 jobs. We represent
# the amount of money each person will produce at each job with a cost matrix.
# Each row corresponds to a person and each column corresponds to a job. So for
# example, below we are saying that person 0 will make $1 at job 0, $2 at job 1,
# and $6 at job 2.
cost = dlib.matrix([[1, 2, 6],
                    [5, 3, 6],
                    [4, 5, 0]])

# To find out the best assignment of people to jobs we just need to call this
# function.
assignment = dlib.max_cost_assignment(cost)

# This prints optimal assignments:  [2, 0, 1]
# which indicates that we should assign the person from the first row of the
# cost matrix to job 2, the middle row person to job 0, and the bottom row
# person to job 1.
print("Optimal assignments: {}".format(assignment))

# This prints optimal cost:  16.0
# which is correct since our optimal assignment is 6+5+5.
print("Optimal cost: {}".format(dlib.assignment_cost(cost, assignment)))