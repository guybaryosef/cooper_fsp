#
# Pset 2: Question 1
# ECE478: Financial Signal Processing
# Brownian Motion
# By: Guy Bar Yosef
#

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


def main():
    m = [1, 5, 10]  # given passage levels
    threshold = 1  # passage time threshold
    walls = [-10, 10]

    estimateProbsAndExps(1000, 10**5, m, threshold)

    graphPathAndReflected(generateBrownainMotion(L = 3), m[1])

    graphWalledPaths(walls[0], walls[1], 5)

    avgTime, avgNum = avgReflectionTimeAndNumber(walls[0], walls[1], 1000)
    prob = probPathInWalledRange(walls[0], walls[1])
    print("Given the walls %d and %d:\n"
            "Average time until first reflection: %f\n"
            "Average number of reflections: %f\n"
            "Probability that path stays inbetween the bounds: %f"% 
                (walls[0], walls[1], avgTime, avgNum, prob))
    


# Part a)
def generateBrownainMotion(N = 10**2, T = 1, L = 1000):
    '''
    Generate L paths of brownian motion from 
    time 0 to time T, with N steps in between.
    Returns a list of a list of the start and end times)
    as well as the steps of the set of paths.

    Note: We are approximating brownian motion using a 
    symmetric random walk.
    '''
    paths = np.zeros([L,N])

    for path in range(L):
        path_val = np.random.binomial(1, 1/2, N-1)

        for step, step_val in enumerate(path_val):
            if step_val:
                paths[path,step+1] = paths[path,step] + 1
            else:
                paths[path,step+1] = paths[path,step] - 1
    return [[0, T], paths]



# Part b)
def firstPassageAndReflection(path, level):
    '''
    Given a brownian motion path and a level, it determines
    the first passage time (returns infinity if level is not
    reached) and the reflected path (with just one relection).

    Note that the paths are inputted as a list of a list of 
    beginning and end times and the actual paths, in steps.
    '''
    first_passage = np.array([np.inf] * path[1].shape[0])
    reflected_path = np.zeros(path[1].shape)

    for path_index, cur_path in enumerate(path[1]):
        for step, cur_path_val in enumerate(cur_path):
            if cur_path_val == level and first_passage[path_index] == np.inf:
                first_passage[path_index] = step * (path[0][1] - path[0][0])/path[1].shape[1]

            if first_passage[path_index] == np.inf:
                reflected_path[path_index, step] = cur_path_val
            else:
                reflected_path[path_index, step] = level - (cur_path_val - level)
    return first_passage, reflected_path



# Part c)
def estimateProbsAndExps(path_num, steps, m, threshold):
    '''
    Given a set of brownian motion paths, each
    with an identical beginning and end time as
    well as step count:
       1. Estimate the probability that the passage time 
    for the set of levels in the list 'm' is greater 
    than time 'threshold'. 
       2. Estimate the conditional expectation of the 
    passage time for the set of levels in the list 'm' 
    given that it is less than time 'threshold'.
    '''
    paths = generateBrownainMotion(N = steps, L = path_num)

    prob = np.empty(len(m)) # the probability
    exp = np.empty(len(m))  # the expectation

    for index, level in enumerate(m):
        passage_time, _ = firstPassageAndReflection(paths, level)

        prob[index] = passage_time[passage_time > threshold].shape[0] / path_num
        exp[index] = np.mean(passage_time[passage_time <= threshold])

    table = Table({'m':m, 'P(Tm > 1)': prob, 'E(Tm|Tm <= 1)': exp})
    print(table)

     

# Part d)
def graphPathAndReflected(paths, m):
    '''
    Given a single level m and a list of brownian motion paths, 
    each with an equal start and end time as well as 
    number of steps, graph both the original and reflected paths.
    '''
    _, reflected_paths = firstPassageAndReflection(paths, m)

    time = np.linspace(paths[0][0], paths[0][1], paths[1].shape[1])
    for count, (org, ref) in enumerate(zip(paths[1], reflected_paths)):
        plt.figure()
        plt.plot(time, org, label='Original Path')
        plt.plot(time, ref, label='Reflected Path')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Brownian Motion Path #%d, m = %d'% (count, m))
        plt.legend()
        plt.show()
        


# Part e)
def reflectedWalls(wall_val1, wall_val2, path_num):
    '''
    Generate path_num number of brownian motion paths
    that are bounded between the two wall values (i.e.
    the path gets reflected at every boundary crossing).

    Note: the initial value of the paths, 0, must be
    between the two inputted wall values.
    '''
    org_paths = generateBrownainMotion(L = path_num)
    bounded_paths = np.zeros(org_paths[1].shape)
    reflection_count = np.zeros(org_paths[1].shape[0])

    for index, (org, bounded) in enumerate(zip(org_paths[1], bounded_paths)):
        in_reflection = 1 # keep track of whether to invert step increments
        for j in range(1, org.shape[0]):
            step_diff = in_reflection*(org[j] - org[j-1] )
            
            if bounded[j-1] + step_diff > wall_val2 or bounded[j-1] + step_diff < wall_val1:
                reflection_count[index] += 1                    # increment reflection count
                in_reflection = -1 * in_reflection    # reflect the step increments
                bounded[j] = bounded[j-1] - step_diff # update bounded path
            else:
                bounded[j] = bounded[j-1] + step_diff

    return reflection_count, org_paths, bounded_paths



def graphWalledPaths(wall_val1, wall_val2, path_num):
    '''
    Given 2 wall values (where 0 has to be in between),
    as well as a number of paths path_num, plot path_num
    plots, each superimposed with their bounded reflected paths
    walled off between the 2 wall values. 
    '''
    _, org, bounded = reflectedWalls(wall_val1, wall_val2, path_num)
    
    time = np.linspace(org[0][0], org[0][1], org[1].shape[1])
    for org_path, bounded_path in zip(org[1], bounded):
        plt.figure()
        plt.plot(time, org_path, label='Original Path')
        plt.plot(time, bounded_path, label='Reflected Path')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Original and Walled Paths, between %d and %d'%(wall_val1, wall_val2) )
        plt.legend()
        plt.show()



def avgReflectionTimeAndNumber(wall_val1, wall_val2, path_num):
    '''
    Given 2 wall bounds (with 0 having to be in between), as well
    as a number of paths path_num, this function returns:
        1. The average time to the first reflection at either wall.
        2. The average number of reflections (at either wall), over
        the span of the paths (which is between 0 and 1, with 1000 steps).
    '''
    reflection_counts, org, _ = reflectedWalls(wall_val1, wall_val2, path_num)

    first_passage = np.array([np.inf]*org[1].shape[0])
    for path_index, cur_path in enumerate(org[1]):
        for step, cur_path_val in enumerate(cur_path):
            if first_passage[path_index] != np.inf:
                break
            if (cur_path_val == wall_val1 or cur_path_val == wall_val2) and first_passage[path_index] == np.inf:
                first_passage[path_index] = step * (org[0][1] - org[0][0])/org[1].shape[1]
                
    return np.mean(first_passage), np.mean(reflection_counts)
    


def probPathInWalledRange(wall_val1, wall_val2):
    '''
    Given a range for a brownian motion path to stay between (must 
    include 0), this function returns the probability that the 
    path stays between the range for its entirety (of course, during
    the allocated time 0<=t<=1, with 10000 steps).
    '''
    reflection_count, _, _ = reflectedWalls(wall_val1, wall_val2, 10**5)
    return reflection_count[reflection_count == 0].shape[0] / reflection_count.shape[0]



if __name__ == '__main__':
    main()