# Pset 3: Question 1
# ECE478: Financial Signal Processing
# Simulating Stochastic Differential Equations
#
# By: Guy Bar Yosef

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table


def main():
    # user defined variables
    path_count = 1000
    N = 250 # path length is (there are roughly 250 work days per year)
    delta = 0.01
    end_time = N*delta
    sigma = 0.1/N
    interest_rate = sigma/3

    # generate geometric brownian motion and print the averate path value at N//2
    GMB_paths = generateGBM(path_count, N, delta, sigma)
    print("Value of S[N/2] is: %f." % np.mean(GMB_paths, axis=0)[N//2])

    # graph 10 of the above geometric brownian motions
    graphPlots(GMB_paths[:10], end_time)

    # show the martingale property at work
    approximateMartingales(GMB_paths[:10], N//2, sigma, delta, interest_rate)


# Part a)
def generateGBM(M, N, delta, sigma):
    '''
    Generates M paths of N-step geometric browinan motion using 
    a Monte Carlo approach.

    The delta signifing step size as well as the variance of 
    the dW increment, and sigma representing the volatility term.
    '''
    alpha = sigma

    GBM_paths = np.ones([M,N])
    for i in range(M):
        for j in range(1, N):
            GBM_paths[i,j] = GBM_paths[i,j-1]*(1 + alpha*delta + sigma*np.random.normal(0, delta))
    
    return GBM_paths



# Part b)
def graphPlots(paths, end_time):
    '''
    Given a set of paths, superimpose the paths 
    on the same graph, with the x-axis representing
    time going from 0 to the inputted end-time.
    '''
    plt.figure()
    time = np.linspace(0, end_time, paths[0].size)
    for i, path in enumerate(paths):
        plt.plot(time, path, label="Path %d"%i)
    
    plt.xlabel("Time")
    plt.title("10 Geometric Brownian Motions")
    plt.legend()
    plt.show()



# Part c & d)
def approximateMartingales(GBM_paths, initial_condition_index, sigma, delta, interest_rate):
    '''
    This function shows the martingale property at work using
    a Monte Carlo approach.

    Given a set of geometric brownian motion paths GBM_paths, we
    begin at the inputted 'initial_condition_index' index and
    simulate for each GBM path 1000 extensions using the
    discounted stock price formula, followed by using the 
    martingale property to estimate the original initial value,
    averaging across the simulated path extensions.

    This function concludes with printing a table comparing the 
    original initial values and the estimated ones, as well as the
    total variance of the estimation.
    '''
    path_count = GBM_paths.shape[0]
    path_len = GBM_paths.shape[1] - initial_condition_index

    original_inital_values = []
    discounted_price_approx = []

    for cur in GBM_paths:
        original_inital_values.append(cur[initial_condition_index])
        paths_discounted_val = np.zeros([1000])

        # find final value and use martingale property to go back to X_hat(initial_condition)
        for i in range(1000):
            x_path_cur_val = cur[initial_condition_index]

            for _ in range(1, path_len):
                x_path_cur_val *= (1 + sigma*np.random.normal(0, delta)) 

            paths_discounted_val[i] = x_path_cur_val*np.exp(path_len*interest_rate)

        discounted_price_approx.append(np.mean(paths_discounted_val))
    
    table = Table({'  Original S[N/2]  ':original_inital_values, \
                   '  Estimated S[N/2]  ': discounted_price_approx})
    print(table)

    print('\nActually, this becomes a fairly trivial issue, because dX = 0*dt + sigma*X*dW '
    'and it is clearly obvious, therefore, that Xi[N] = Xi[N/2] + Y where Y is a random '
    'variable that is independent of the filtration up to time N/2 and (under the risk-'
    'neutral measure) has mean value 0, so obviously'
    'E_tilda[ X[N] | N/2 ] = X[N/2].\n')

    variance_estimation = 0
    for org, est in zip(original_inital_values, discounted_price_approx):
        variance_estimation += (org-est)**2
    print('The estimation for THE VARIANCE is %8f'% (variance_estimation*(1/path_count)))



if __name__ == '__main__':
    main()