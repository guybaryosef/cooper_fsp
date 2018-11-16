# Pset 3: Question 2
# ECE478: Financial Signal Processing
# Interest Rate Models
#
# By: Guy Bar Yosef

import numpy as np
import matplotlib.pyplot as plt


def main():
    # defined in question
    N = 250
    end_time = 5

    # interest rate constants [Ka, Kb, Kc]
    K = [0.1, 0.3, 0.8]
    
    #lambda functions for interest rate parameters
    b_t = lambda t: K[1]*(1.1 + np.sin(np.pi*t/end_time))
    sigma_t = lambda t: K[2]*(1.1 + np.cos(4*np.pi*t/end_time))
    a_t = lambda t: (1/2)*sigma_t(t)**2 + K[0]*(1.1 + np.cos(np.pi*t/end_time))

    # preliminary analysis of the lambda functions above
    plotParameters(end_time, b_t, sigma_t, a_t, end_time)

    # simulating the interest rate paths, counting the occuring exceptions
    paths_HW, paths_CIR, exceptions = simulateInterestRate(b_t, sigma_t, a_t, endTime=end_time)

    # plotting the simulated insurance rate paths
    plottingInterestRatePaths([paths_HW, paths_CIR], end_time)

    # finding the mean and variance at each point
    findMeanVariance(b_t, sigma_t, a_t, M=1000, N=250, end_time=end_time)

    # part d)
    print('There were %d exceptions for R_HW and %d exceptions for R_CIR,'
            ' over %d total steps.\n'% (exceptions[0], exceptions[1], N))



# part b)
def simulateInterestRate(b_t, sigma_t, a_t, M=10, N=250, endTime=1):
    '''
    Returns M simulated random paths, each of N steps, of the 
    Hull-White and Cox-Ingersoll-Ross interest rate models
    using the b_t, sigma_t, and a_t inputted parameters.

    It also returns the number of occuring exceptions for each of these
    models. An exception occurs whenever the interest rate value
    'approaches' 0, as it can't technically reach it in real life. 
    We take the epsilon to be 0.01 in this case.
    '''
    # we consider it an exception if R gets 'close' to 0
    exception_counts = np.zeros([2])
    exception_tolerance = 0.01

    paths_HW = np.ones([M, N])
    paths_CIR = np.ones([M, N])

    time = np.linspace(0, endTime, N)
    for j,t in zip(range(1,N), time[1:]):
        for i in range(M):

            # iterate the interest rates, keeping count of the exceptions
            paths_HW[i,j] = paths_HW[i,j-1] + (a_t(t) - b_t(t)*paths_HW[i,j-1])*(endTime/N) + \
                                                sigma_t(t)*np.random.normal(0, endTime/N)
            if paths_HW[i,j] <= exception_tolerance:
                exception_counts[0] += 1
                paths_HW[i,j] = exception_tolerance

            paths_CIR[i,j] = paths_CIR[i,j-1] + (a_t(t) - b_t(t)*paths_CIR[i,j-1])*(endTime/N) + \
                                                sigma_t(t)*np.random.normal(0, endTime/N)

            if paths_CIR[i,j] <= exception_tolerance:
                exception_counts[1] += 1
                paths_CIR[i,j] = exception_tolerance

    return paths_HW, paths_CIR, exception_counts



def plottingInterestRatePaths(list_of_interest_rates, end_time=1):
    '''
    Given a list of interest rate paths, this function graphs them
    over time, going from 0 to the specified end_time.
    '''
    time = np.linspace(0, end_time, list_of_interest_rates[0].shape[1])
    plt.figure()
    for i, path in enumerate(list_of_interest_rates[0]):
        plt.plot(time, path, label='Path #%d'%(i+1))
    plt.title('10-path Simulation of $R^{HW}$')
    plt.xlabel('time')
    plt.ylabel('$R^{HW}$')
    plt.legend()
    plt.show()

    plt.figure()
    for i, path in enumerate(list_of_interest_rates[1]):
        plt.plot(time, path, label='Path #%d'%(i+1))
    plt.title('10-path Simulation of $R^{CIR}$')
    plt.xlabel('time')
    plt.ylabel('$R^{CIR}$')
    plt.legend()
    plt.show()

    print('\nThe K parameters were chosen so that the interest rate neither'
        ' blows up or dies down, rather staying relatively close to its'
        ' inital value (int this case, 1).\nThe epsilon value was picked'
        ' to be close enough to zero so as to avoid constant exceptions,'
        ' yet high enough so that the interest rate would never actually'
        ' reach 0.\n')



# part c)
def findMeanVariance(b_t, sigma_t, a_t, M=1000, N=250, end_time=1):
    '''
    Using a Monte Carlo approach, this finds the
    mean and variance of each interest rate model at every
    step in the N-step path, from time 0 to the inputted end_time.
    '''
    paths_HW, paths_CIR, _ = simulateInterestRate(b_t, sigma_t, a_t, M, N, end_time)
    expected_values = [np.mean(paths_HW, axis=0), np.mean(paths_CIR, axis=0)]
    variances = [np.var(paths_HW, axis=0), np.var(paths_CIR, axis=0)]

    time = np.linspace(0, end_time, N)
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(time , expected_values[0], label='Hull-White')
    plt.plot(time , expected_values[1], label='Cox-Ingersoll-Ross')
    plt.title('Expected Value of the Interest Rates')
    plt.xlabel('time')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(time , variances[0], label='Hull-White')
    plt.plot(time , variances[1], label='Cox-Ingersoll-Ross')
    plt.title('Variance of the Interest Rates')
    plt.xlabel('time')
    plt.legend()
    plt.show()

    return expected_values, variances



# preliminary analysis
def plotParameters(T, b_t, sigma_t, a_t, end_time):
    '''
    Plots the three interest rate parameters b_t,
    sigma_t, and a_t, over the inputted time interval. 
    '''
    time = np.linspace(0, end_time, 1000)

    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(time, b_t(time))
    plt.title('$b_t$ parameter')

    plt.subplot(1,3,2)
    plt.plot(time, sigma_t(time))
    plt.title('$sigma_t$ parameter')

    plt.subplot(1,3,3)
    plt.plot(time, a_t(time))
    plt.title('$a_t$ parameter')
    plt.show()



if __name__ == '__main__':
    main()