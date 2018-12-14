# ECE:478 Financial Signal Processing
# Pset 4: Financial Time-Series Analysis
# By: Guy Bar Yosef
#
# Q2 - Cointegration of AR models


import preprocess as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

    # get data for the user-specified year
    year = input('Input year to test between 2000 and 2016: ')
    sp_daily_returns = pd.read_csv('./by_years/SP_daily_returns_'+year+'.csv')
    ff48_daily_returns = pd.read_csv('./by_years/48_IP_eq_w_daily_returns_'+year+'.csv')

    print('Checking the cointegratedness of Coal and Oil stocks:')
    r1 = ff48_daily_returns['Coal ']
    r2 = ff48_daily_returns['Oil  ']
    delta1, w1 = ls_fitting(r1)
    delta2, w2 = ls_fitting(r2)
    delta12, w12 = ls_fitting(r2, r1)
    
    print('delta 1:', delta1, '\tweight 1:', w1)
    print('delta 2:', delta2, '\tweight 2:', w2)
    print('delta 12:', delta12, '\tweight 12:', w12)

    lag_count = 50 # user defined
    lags = np.arange(0, lag_count, 1)
    gammas = cov_coeff(r2, r1, delta12, w12, lag_count)
    plt.figure()
    plt.stem(lags, gammas)
    plt.xlabel('Lag')
    plt.ylabel('Covariance')
    plt.title('Covariance of a pair of benchmarked assets')
    plt.savefig('./outputs/FSP_pset4_q2_benchmark_assets.pdf')

    print('\nSimulating cointegrated time series:')
    co_r1, co_r2 = generate_cointegrated(0.5, 0.5) # just generate it ourselves
    co_delta1, co_w1 = ls_fitting(co_r1)
    co_delta2, co_w2 = ls_fitting(co_r2)
    co_delta12, co_w12 = ls_fitting(co_r2, co_r1)
    
    print('cointegrated delta 1:', co_delta1, '\tcointegrated weight 1:', co_w1)
    print('cointegrated delta 2:', co_delta2, '\tcointegrated weight 2:', co_w2)
    print('cointegrated delta 12:', co_delta12, '\tcointegrated weight 12:', co_w12)

    co_gammas = cov_coeff(co_r2, co_r1, co_delta12, co_w12, lag_count)
    plt.figure()
    plt.stem(lags, co_gammas)
    plt.xlabel('Lag')
    plt.ylabel('Covariance')
    plt.title('Covariance of a pair of cointegrated Assets')
    plt.savefig('./outputs/FSP_pset4_q2_cointegrated_assets.pdf')



def ls_fitting(x_in, y_in=None):
    '''
    Performs a least squares fitting to the data r
    of an AR(1) model, with drift:
        r_{1t} = delta + w_1*r_{1(t-1)} + v, where v is iid gaussian.
    Returns the drift and weight values.
    '''
    if y_in is not None:
        y = np.transpose(np.array(y_in, ndmin=2))
        x = np.transpose(np.array(x_in, ndmin=2))
    else:
        y = np.transpose(np.array(x_in[1:], ndmin=2))
        x = np.transpose(np.array(x_in[:-1], ndmin=2))

    x = np.insert(x, 0, 1, axis=1)
    beta = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y

    return beta[0][0], beta[1][0]



def cov_coeff(r2, r1, delta12, w12, lag_count):
    '''
    Finds the covariance coefficients of the error term
    of the AR(1) model predicting r2 from r1, with weight
    w12 and delta term delta12, with the lag ranging from
    0 to 20.
    '''
    e = r1 - delta12 - w12*r2
    e_mean = np.mean(e)
    gamma = []
    for k in range(lag_count):
        cur = 0
        for n in range(e.size-k-1):
            cur += (e[n+k]-e_mean)*(e[n]-e_mean)
        cur *= 1/(e.size-k)

        gamma.append(cur)
    return gamma
    


def generate_cointegrated(alpha1, alpha2):
    '''
    Generates 2 unit-root nonstationary time series, x and y,
    that are cointegrated. alpha1 and alpha2 are the
    coefficients of x and y respectively in the stationary 
    linear combination of the two time series.
    '''
    x = np.ones([250])
    y = np.ones([250])

    for i in range(1, 250):
        x[i] = x[i-1] - 0.8*(x[i-1]+(alpha2/alpha1)*y[i-1]) + np.random.normal(0,0.2) 
        y[i] = y[i-1] - 0.8*(y[i-1]+(alpha1/alpha2)*x[i-1]) + np.random.normal(0,0.2) 

    return x, y



if __name__ == '__main__':
    main()