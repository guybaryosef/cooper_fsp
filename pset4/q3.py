# ECE:478 Financial Signal Processing
# Pset 4: Financial Time-Series Analysis
# By: Guy Bar Yosef
#
# Q3 - ARMA-GARCH models


import preprocess as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.univariate import arch_model


def main():

    # get data for the user-specified year
    year='2000'
    #year = input('Input year to test between 2000 and 2016: ')
    sp_daily_returns = pd.read_csv('./by_years/SP_daily_returns_'+year+'.csv')
    ff48_daily_returns = pd.read_csv('./by_years/48_IP_eq_w_daily_returns_'+year+'.csv')

    r = ff48_daily_returns['Coal ']
    
    # User specified coefficients:
    # arma coefficients
    b = [0.2]
    a = [0.4]
    # garch coefficients
    c = [0.2, 0.3]
    d = [0.4]

    # part a)
    print('First with a gaussian distribution:')
    r = generateARMA_GARCH(a, b, c, d, 250)

    # part b)
    b_est, a_est = fit_ARMA_to_stock(r)
    print(b_est, a_est)

    # part c)
    get_GARCH_coeff(r)

    # part d)
    print('Now with a student-t distribution:')
    r2 = generateARMA_GARCH(a, b, c, d, 250, 'student t')
    b_est2, a_est2 = fit_ARMA_to_stock(r)
    print(b_est2, a_est2)
    get_GARCH_coeff(r2)
    


# part a & e)
def generateARMA_GARCH(a, b, c, d, step_count, distribution='gaussian'):
    '''
    Given the ARMA(1,1) coefficients b and a and the GARCH(1,1)
    coefficients c and d, generate a fictitious stock
    with step_count number of steps with initial value 0. 
    
    The noise in the GARCH model will be either gaussian or following a
    student-t distribution, depending on the inputted distribution parameter.
    '''
    # GARCH model helps us model the volatility of the ARMA model:
    sigma = np.zeros([step_count])
    v = np.zeros([step_count])
    fictitious_stock = np.zeros([step_count]) 
    for i in range(1, step_count):

        # GARCH simulation
        sigma[i] = np.sqrt(c[0] + c[1]*v[i-1]**2 + d[0]*sigma[i-1]**2)
        
        if distribution == 'gaussian':
            v[i] = sigma[i]*np.random.normal(0, 0.2)
        else:
            v[i] = sigma[i]*np.random.standard_t(8)

        # ARMA simulation
        fictitious_stock[i] = b[0]*fictitious_stock[i-1] - a[0]*v[i-1] + v[i]
    
    n = np.arange(0, 250)
    plt.figure()
    plt.plot(n, sigma, label='Sigma')
    plt.plot(n, v, label='v')
    plt.plot(n, fictitious_stock, label='r')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Value')
    plt.title('Signal Components')
    plt.savefig('./outputs/FSP_tsa_q3_'+distribution)

    return fictitious_stock



# part b)
def fit_ARMA_to_stock(r):
    '''
    Given a stock, this function fits an ARMA(1,1)
    model to it using least squares.

    It returns (in the presented order) the AR and MA
    coefficients.
    '''
    # least squares set-up
    y = np.transpose(np.array(r[1:], ndmin=2))
    x = np.transpose(np.array(r[:-1], ndmin=2))
    b = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y

    ar_residual = y - x @ b
    
    a = np.sum(ar_residual)

    return b[0,0], a



# part c)
def get_GARCH_coeff(r):
    '''
    Finds the GARCH parameters.
    '''
    am = arch_model(r)
    tmp = am.fit()
    print(tmp.summary())



if __name__ == '__main__':
    main()