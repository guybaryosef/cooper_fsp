# ECE:478 Financial Signal Processing
# Pset 4: Financial Time-Series Analysis
# By: Guy Bar Yosef
#
# Q1) Autoregressive models and Levinson-Durbin

import preprocess as pp
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import toeplitz
from statsmodels.tsa import stattools
from statsmodels.tsa.ar_model import AR
from astropy.table import Table


def main():

    pp.preprocess()

    # user defined values
    M = 10 # order of the AR model
    do_it_all(M)



def do_it_all(M):

    # get data for the user-specified year
    year = input('Input year to test between 2000 and 2016: ')
    sp_daily_returns = pd.read_csv('./by_years/SP_daily_returns_'+year+'.csv')
    ff48_daily_returns = pd.read_csv('./by_years/48_IP_eq_w_daily_returns_'+year+'.csv')
    
    r1 = ff48_daily_returns['Agric']

    # part a)
    cov_lag_mat = covLagMat(M, r1)
    eig_vals, _ = np.linalg.eig(cov_lag_mat)
    print('Eigenvalues of the lag-covariance matrix:\n', eig_vals)

    # part b)
    print('ACTUAL TIME SERIES RESULTS:')
    estimate_ar_coefficients(r1, 10)

    print('The Reflection coefficients seem to die out after roughly ',
    'the first one, so it seems that the best model is AR(1).')

    # part c)
    min_order = AIC_calc(r1, 10)
    print('Model order with the smalles AIC value:', min_order, end='\n\n\n')

    # part d)
    print('FIRST-ORDER DIFFERENCE RESULTS:')
    first_order_sys(r1, M)

    # part e)
    covRefOrdersFun(r1, min_order, M)

    # part f) 
    # NOTE: This function can only do cumulants of order 1-4
    kurtosis, cumulants = find_kurtosis_cumulants(r1)
    print('\nKurtosis:', kurtosis)
    print('Cumulants or order 1-4:', cumulants)



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
        gamma_cur = 0
        for n in range(r.size-k-1):
            gamma_cur += (r[n+k]-r_mean)*(r[n]-r_mean)
        gamma_cur *= 1/(r.size-k)
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
    
    print('\nLevinson Durbin Reflection coefficients:', -reflec_coeff)
    print('Levinson-durbin delta:', delta_0)

    # Least squares: (the actual regression is needed for the delta term)
    y = np.transpose(np.array(r[M:], ndmin=2))
    x = np.zeros([r.size-M, M])
    x = np.insert(x, 0, 1, axis=1) # for the delta term (y-intercept)
    for i in range(M, r.size):
        for j in range(1, M+1):
            x[i-M, j] = r[i-j]
        
    beta = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y

    ols_pacf = stattools.pacf_ols(r, M)
    print('\nLeast Squares Reflection coefficients:', -1*ols_pacf) 
    print('Least Squares delta:', np.sum(beta[1:])*r.mean(), end='\n\n')

    return np.sum((y - x @ beta)**2)/y.size



# part c)
def AIC_calc(r, M):
    '''
    Calculates the AIC values for orders 1 through input M
    and finds the best order (minimum value), returning
    the optimal model order.
    '''
    best_val = 100000 # some arbitrary ceiling value 
    best_order = 0
    for i in range(1, M+1):
        # perform least squares
        y = np.transpose(np.array(r[i:], ndmin=2))
        x = np.zeros([r.size-i, i])
        x = np.insert(x, 0, 1, axis=1) # for the delta term (y-intercept)
        for k in range(i, r.size):
            for j in range(1, i+1):
                x[k-i, j] = r[k-j]
        
        beta = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y
        
        # calculate the AIC value of the model using the error of the least squares
        cur_aic_val = (2*(i+2)/r.size) + np.log(np.sum((y - x @ beta)**2)/y.size)

        if cur_aic_val < best_val:
            best_val = cur_aic_val
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

    print('Model order with the smallest AIC:', AIC_calc(c, M), end='\n\n')
    return c, AIC_calc(c, M)



# part e)
def covRefOrdersFun(r, org_opt_ord, M):
    '''
    '''
    org_orders = [1, org_opt_ord, 10] # orders to check for original data

    c = []
    for i in range(1, r.size):
        c.append(r[i] - r[i-1])
    c = pd.Series(c)
    fd_opt_ord = AIC_calc(c, M)
    fdiff_orders = [1, fd_opt_ord, 10] # orders to check for 1st ord. diff

    data = [r, c]
    for ts in [r,c]:
        cov_coeff_total = []
        ref_coeff_total = []
        for order in org_orders:

            y = np.transpose(np.array(ts[order:], ndmin=2))
            x = np.zeros([ts.size-order, order])
            x = np.insert(x, 0, 1, axis=1) # for the delta term (y-intercept)
            for k in range(order, ts.size):
                for j in range(1, order+1):
                    x[k-order, j] = ts[k-j]
            
            beta = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y

            v = y - x @ beta

            v_mean = v.mean()
            gamma = []
            for k in range(M+1):
                gamma_cur = 0
                for n in range(v.size-k-1):
                    gamma_cur += (v[n+k]-v_mean)*(v[n]-v_mean)
                gamma_cur *= 1/(v.size-k)
                gamma.append(gamma_cur)

            cov_coeff_total.append(gamma)
            
            reflec_coeff = []
            for i in range(M+1):
                reflec_coeff.append(gamma[i]/gamma[0])

            ref_coeff_total.append(reflec_coeff)
            
            if np.array_equal(data, r):
                title = 'Original Data'
            else:
                title = '1st Order Difference'
        table = Table({
                    title : range(M+1), \
                    'Cov Coeff, Ord 1': cov_coeff_total[0],  \
                    'Ref Coeff, Ord 1': ref_coeff_total[0], \
                    'Cov Coeff, Ord'+str(org_opt_ord): cov_coeff_total[1], \
                    'Ref Coeff, Ord'+str(org_opt_ord): ref_coeff_total[1], \
                    'Cov Coeff, Ord 10':  cov_coeff_total[2], \
                    'Ref Coeff, Ord 10':  ref_coeff_total[2], \
                    })
        print(table)

    print('\nThe redisual error term, v, would be white if the covariance coefficients',
          'become 0 for orders greater than the model order. While not absolute, there',
          'does appear to be a declining trend and so I would say... they are kind-of white...')



# part f)
def find_kurtosis_cumulants(r):
    '''
    Finds the Kurtosis of the inputted vector, r, as
    well as its cumulants of orders 1 through 4.
    '''
    y = np.transpose(np.array(r[10:], ndmin=2))
    x = np.zeros([r.size-10, 10])
    x = np.insert(x, 0, 1, axis=1) # for the delta term (y-intercept)
    for k in range(10, r.size):
        for j in range(1, 10+1):
            x[k-10, j] = r[k-j]

    beta = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y

    v = y - x @ beta

    kurtosis = (((v-v.mean())**4)/(np.var(v)**2)).mean()
    cumulants = []
    for i in range(1, 5):
        cumulants.append(stats.kstat(v, i))
    return kurtosis, cumulants



if __name__ == '__main__':
    main()