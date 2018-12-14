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
    year = input('Input year to test between 2000 and 2016: ')
    sp_daily_returns = pd.read_csv('./by_years/SP_daily_returns_'+year+'.csv')
    ff48_daily_returns = pd.read_csv('./by_years/48_IP_eq_w_daily_returns_'+year+'.csv')

    r = ff48_daily_returns['Coal ']
    
    # User specified coefficients:
    sigma_v = 1
    # arma coefficients
    b = [3]
    a = [0.2]
    # garch coefficients
    c = [0.1, 0.3]
    d = [0.2]

    # part a)
    print('First with a gaussian distribution:')
    r = generateARMA_GARCH(a, b, c, d, 250, sigma_v)

    # part b)
    b_est, a_est, v_est = fit_ARMA_to_stock(r)
    print('b[0] =', b_est, '\ta[0] = ', a_est, end='\n\n')
    
    # part c)
    get_GARCH_coeff(v_est)

    # part d)
    print('\nNow with a student-t distribution:')
    r2 = generateARMA_GARCH(a, b, c, d, 250, sigma_v, 'student t')
    b_est2, a_est2, v_est2 = fit_ARMA_to_stock(r2)
    print('b[0] =', b_est2, '\ta[0] = ', a_est2, end='\n\n')

    get_GARCH_coeff(v_est2)
    


# part a & e)
def generateARMA_GARCH(a, b, c, d, step_count, vsigma=1, distribution='gaussian'):
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
            v[i] = sigma[i]*np.random.normal(0, vsigma)
        else:
            v[i] = sigma[i]*np.random.standard_t(8)

        # ARMA simulation
        fictitious_stock[i] = b[0]*fictitious_stock[i-1] + v[i] - a[0]*v[i-1]

    n = np.arange(0, 250)
    plt.figure()
    plt.plot(n, sigma, label='Sigma')
    plt.plot(n, v, label='v')
    plt.plot(n, fictitious_stock, label='r')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Value')
    plt.title('Signal Components')
    plt.savefig('./outputs/FSP_tsa_q3_'+distribution+'.pdf')

    return fictitious_stock



# part b)
def fit_ARMA_to_stock(r):
    '''
    Given a stock, this function fits an ARMA(1,1)
    model to it using least squares.

    The input v_sigma is the variance for the signal v,
    the residual error.

    This function returns the AR and MA coefficients.
    '''
    # find the b coefficient
    y = np.transpose(np.array(r[1:], ndmin=2))
    x = np.transpose(np.array(r[:-1], ndmin=2))
    b = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y 

    # find the a coefficient
    # calculate the covariance of r of lags 0 and 1
    residuals = y - x @ b
    residuals_mean = residuals.mean()
    gamma_1 = 0
    gamma_0 = 0
    for i in range(residuals.size-1):
        gamma_1 += (residuals[i+1,0]-residuals_mean)*(residuals[i,0]-residuals_mean)
        gamma_0 += (residuals[i,0]-residuals_mean)*(residuals[i,0]-residuals_mean)
    gamma_0 += (residuals[-1,0]-residuals_mean)*(residuals[-1,0]-residuals_mean)

    gamma_1 *= (1/(residuals.size-2))
    gamma_0 *= (1/(residuals.size-1))
    
    corr_coef = gamma_1/gamma_0
    
    a = (1+np.sqrt(1-4*corr_coef**2))/(2*corr_coef)

    # find v, the residual error of our data vector r
    v = np.zeros([r.size])
    for i in range(1, r.size-1):
        v[i] = residuals[i] + a*v[i-1]

    return b[0,0], a, v



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