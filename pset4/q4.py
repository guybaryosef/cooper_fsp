# ECE:478 Financial Signal Processing
# Pset 4: Financial Time-Series Analysis
# By: Guy Bar Yosef
#
# Q4 - Time Series Simulation


import preprocess as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():

    pp.preprocess()

    # get data for the user-specified year
    year = input('Input year to test between 2000 and 2016: ')
    sp_daily_returns = pd.read_csv('./by_years/SP_daily_returns_'+year+'.csv')
    ff48_daily_returns = pd.read_csv('./by_years/48_IP_eq_w_daily_returns_'+year+'.csv')

    x = np.array(ff48_daily_returns[['Coal ', 'Oil  ']])

    # part a)
    # we are able to do this because of our assumption that v is white
    eyda1 =np.array([1, -0.7, 0, -0.6])
    eyda2 = np.array([0, -0.5, 1, -0.7])
    corr_coef = np.sum(eyda1*eyda2)/np.sqrt(np.sum(eyda1**2)*np.sum(eyda2**2))
    print('Correlation coefficient of the eta\'s is:', corr_coef)

    # part b)
    _, _, y1, y2 = simulateVARMA()

    # part c)
    lag_cov_y1 = test4stationarity(y1)
    lag_cov_y2 = test4stationarity(y2)
    print('\ny1 lag covariance:', lag_cov_y1)
    print('y2 lag covariance:', lag_cov_y2)

    print('\nThe covariances of y1 do not decay in the same magnitude',
        'as the covariances of y2, and so indicate that y2 is stationary ',
        'while y1 is not.')

    # part d)
    print('\nLooking at the graphs, we can see that y1 closely follows x1, giving',
        'it a non-zero mean and therefore making it not stationary.', 
        'Meanwhile y2 seems to exhibit variations around the value 0, hinting', 
        'that it has 0 mean.')



# part b)
def simulateVARMA():
    '''
    Given the specifications in the problem, simulate the x1, x2 and
    y1, y2 stocks in the VARMA(1,1) model, assuming v is zero-mean 
    with covariance matrix I.
    '''
    n = np.arange(0, 250)
    v_sets = np.random.normal(0, 1, [5, 250, 2])

    x = np.zeros([250,2])
    y = np.zeros([250, 2])

    for rep in range(5):
        for i in range(1, 250):   # simulate x1 and x2
            x[i,0] = 0.5*x[i-1,0] - x[i-1,1] + v_sets[rep,i,0] - 0.2*v_sets[rep,i-1,0] + 0.4*v_sets[rep,i-1,1]
            x[i,1] = 0.5*x[i-1,1] - 0.25*x[i-1,0] + v_sets[rep,i,1] + 0.1*v_sets[rep,i-1,0] - 0.2*v_sets[rep,i-1,1]

        for i in range(2, 250):   # simulate y1, and y2
            y[i,0] = y[i-1,0] + (x[i,0]-x[i-1,0]) - 0.4*(x[i-1,0]-x[i-2,0])
            y[i,1] = (x[i,1]-x[i-1,1])
        
        plt.figure()
        plt.plot(n, x[:,0], label='x1')
        plt.plot(n, x[:,1], label='x2')
        plt.plot(n, y[:,0], label='y1')
        plt.plot(n, y[:,1], label='y2')
        plt.title('Time Series Simulation, Rep #%d'%(rep+1))
        plt.xlabel('Time Steps')
        plt.ylabel('Signal Value')
        plt.legend()
        plt.savefig('./outputs/FSP_pset4_q4_b_rep'+str(rep+1)+'.pdf')

    return x[:,0], x[:,1], y[:,0], y[:,1]



# part c)
def test4stationarity(y):
    '''
    Given a time series, print out its lag covariance
    for lags 0 through 10. 
    
    Note that decaying covariances (with increasing lag
    time) is a sign of stationarity. 
    '''
    y_mean = y.mean()
    lag_cov = np.zeros([11])
    
    for k in range(0,11):
        tmp_sum = 0
        for i in range(k, y.size):
            tmp_sum += (y[i]-y_mean)*(y[i-k]-y_mean)

        lag_cov[k] = (1/(y.size))*tmp_sum

    return lag_cov



if __name__ == '__main__':
    main()