# ECE:478 Financial Signal Processing
# Pset 4: Financial Time-Series Analysis
# By: Guy Bar Yosef
#
# Q1) Autoregressive models and Levinson-Durbin

import preprocess as pp
import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from statsmodels.tsa import stattools
from statsmodels.tsa.ar_model import AR

def main():

    #pp.preprocess()

    # user defined values
    M = 10 # order of the AR model
    do_it_all(M)



def do_it_all(M):

    # get data for the user-specified year
    year = '2000'
    #year = input('Input year to test between 2000 and 2016: ')
    sp_daily_returns = pd.read_csv('./by_years/SP_daily_returns_'+year+'.csv')
    sp_daily_returns = sp_daily_returns['Return']

    # part a)
    cov_lag_mat = covLagMat(M, sp_daily_returns)
    eig_vals, _ = np.linalg.eig(cov_lag_mat)
    print('Eigenvalues of the lag-covariance matrix:\n', eig_vals)

    # part b)
    estimate_ar_coefficients(sp_daily_returns, M)
    print('The Reflection coefficients seem to die out after roughly ',
    'the first one, so it seems that the best model is AR(1).')

    # part c)
    min_order = AIC_calc(sp_daily_returns, M)
    print('Model order with the smalles AIC value:', min_order, end='\n\n')

    # part d)
    print('First order difference data:')
    first_order_sys(sp_daily_returns, M)

    # part e)

    # part f) (not taking the kurtosis of the right thing...)
    print('Kurtosis:', find_kurtosis(sp_daily_returns))



# Part a)
def covLagMat(M, r):
    '''
    Given a vector r and an order M less than
    or equal to the length of r, returns the 
    Toeplitz covariance matrix with lag from 
    0 to M.
    '''
    r_mean = r.mean()
    gamma = []
    for k in range(M+1):
        gamma_cur = 1/(r.size-k)
        for n in range(r.size-k-1):
            gamma_cur += (r[n+k]-r_mean)*(r[n]-r_mean)
        gamma.append(gamma_cur)
    return toeplitz(gamma)



# Part b)
def estimate_ar_coefficients(r, M):
    '''
    Runs Levinson-Durbin to compute the AR coefficients,
    reflection coefficients, and prediction error powers up to M.
    This function then uses a least squares fit to compute the 
    reflection coefficients from the AR coefficients and compares
    the results.
    '''
    # Levinson Durbin:
    _, ar_coeff, reflec_coeff, _, _ = stattools.levinson_durbin(r, M)
    delta_0 = r.mean()*np.sum(ar_coeff)
    
    print('\nLevinson Durbin Reflection coefficients:', reflec_coeff)
    print('Levinson-durbin delta:', delta_0)

    # Least squares: (the actual regression is needed for the delta term)
    y = np.transpose(np.array(r[M:], ndmin=2))
    x = np.zeros([r.size-M, M])
    x = np.insert(x, 0, 1, axis=1) # for the delta term (y-intercept)
    for i in range(M, r.size):
        x[i-M, 1] = r[i-1]
        x[i-M, 2] = r[i-2]
        x[i-M, 3] = r[i-3]
        x[i-M, 4] = r[i-4]
        x[i-M, 5] = r[i-5]
        x[i-M, 6] = r[i-6]
        x[i-M, 7] = r[i-7]
        x[i-M, 8] = r[i-8]
        x[i-M, 9] = r[i-9]
        x[i-M, 10] = r[i-10]

    beta = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y
    
    ols_pacf = stattools.pacf_ols(r, M)
    print('\nLeast Squares Reflection coefficients:', ols_pacf) 
    print('Least Squares delta:', beta[0, 0], end='\n\n')



# part c)
def AIC_calc(r, M):
    '''
    Calculates the AIC values for orders 1 through input M
    and finds the best order (minimum value), returning
    the optimal model order.
    '''
    _, _, _, prediction_error, _ = stattools.levinson_durbin(r, M)
    print(prediction_error)
    best_val = 100000
    best_order = 0
    for i in range(1, M+1):
        
        cur_val = (2/r.size)*(-np.log(np.sum(prediction_error[i]**2)) + i)
        print(cur_val)
        if cur_val < best_val:
            best_val = cur_val
            best_order = i
    
    return best_order



# part d)
def first_order_sys(r, M):
    '''
    Finds the first-order difference of the data and 
    computes its covariance coefficients as well as its
    AR and reflection coefficients using levinson
    durbin recusion.
    '''
    c = []
    for i in range(1, r.size):
        c.append(r[i] - r[i-1])
    c = pd.Series(c)
    fos_cov = covLagMat(M, r)

    estimate_ar_coefficients(c, M)

    print('1st order difference model order with the smallest AIC:', AIC_calc(c, M), end='\n\n')



# part e)




# part f)
def find_kurtosis(r):
    '''
    Finds the Kurtosis of the inputted vector, r.
    '''
    return (((r-r.mean())**4)/(np.var(r)**2)).mean()




if __name__ == '__main__':
    main()