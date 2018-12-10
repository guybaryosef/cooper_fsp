# 
# Pset 1
# ECE 478 Financial Signal Processing
# Sparse Portfolio Analysis
# 
# By: Guy Bar Yosef


import numpy as np
import pandas as pd
import q1  # question 1
import copy
import cvxpy

def main():

    # get data for the user-specified year
    year = 2000
    #year = input('Input year to test between 2000 and 2016: ')
    
    # part a)
    # weights of 3 randomly chosen points on the efficient frontier
    weights = getWeights(year)
    
    l1_weights, l1_k = sparcifyWeights(weights, 1)
    l2_weights, l2_k = sparcifyWeights(weights, 2)

    print('0-norm of S1:', l1_k)
    print('0-norm of S2:', l2_k)

    org_row, org_sigma = row_sigma(weights, year)
    l1_row, l1_sigma = row_sigma(l1_weights, year)
    l2_row, l2_sigma = row_sigma(l2_weights, year)
    
    print('Orig. Sharpe ratios :', org_row/org_sigma)
    print('Sharpe ratios for S1:', l1_row/l1_sigma)
    print('Sharpe ratios for S2:', l2_row/l2_sigma)

    # part b)
    tau = 0.5
    row = 0.5
    data = pd.read_csv('./by_years/48_IP_eq_w_daily_returns_'+str(year)+'.csv').values[:,1:]
    
    lasso_weights = optimizedLassoProblem(data, row, tau)
    lasso_row, lasso_sigma = row_sigma(lasso_weights, year)



# part a)
def sparcifyWeights(weights, p):
    '''
    Given a set of weights (may be 2-d) and an order p,
    this function returns the sparsified portfolios under
    the p-norm, as well as the optimal number of non-zero
    coefficients (weights). 
    '''
    sparce_weights = weights
    k_val = []
    for i in range(weights.shape[1]):
        norm_val = np.linalg.norm(weights[:,i], ord=p)
        min_val = norm_val
        best_val = 48
        best_weights = []

        for k in range(1, weights[:,i].size):
            tmp = copy.deepcopy(weights[:,i])
            count_to_zero = weights[:,i].size - k
            
            while count_to_zero > 0:
                to_zero = np.union1d(np.where(abs(tmp) == abs(tmp).min()), np.where(abs(tmp) > 0))
                if count_to_zero < to_zero.size:
                    to_zero = to_zero[:count_to_zero]
                for j in to_zero:
                    tmp[j] = 0
                count_to_zero -= to_zero.size
            
            tmp = tmp/np.sum(tmp)
            tmp_norm = np.linalg.norm(weights[:,i] - tmp, ord=p)
            if tmp_norm < min_val:
                min_val = tmp_norm
                best_val = k
                best_weights = tmp
            
        sparce_weights[:,i] = best_weights
        k_val.append(best_val)

    return sparce_weights, k_val



def row_sigma(portfolio, year):
    '''
    Given a set of weights and a year (between 2000-2016),
    this function finds the expected return and risk of the
    ff48 benchmark portfolio with the given weights.
    '''
    meanVec, covMat = q1.find_mean_cov(year)    

    row = np.zeros([portfolio.shape[1]])
    sigma = np.zeros([portfolio.shape[1]])

    for i in range(portfolio.shape[1]):
        row[i] = (np.transpose(meanVec) @ portfolio[:,i])[0]
        sigma[i] =  np.sum(np.sqrt( np.transpose(portfolio[:,i]) @ covMat.values @ portfolio[:,i] ))
    return row, sigma



def getWeights(year):
    '''
    Given a year, this function returns a numpy array of 
    the weights of the following portoflios: naive portfolio,
    3 points on the efficient frontier, MVP, and the market 
    portfolio.
    '''    
    # naive portofolio weights
    naive_weights = pd.DataFrame({'Weights': ([1/48] * 48)})

    #MVP weights
    meanVec, covMat = q1.find_mean_cov(year)    
    sigma_mvp_portfolio = 1/np.sqrt( np.sum(np.linalg.inv(covMat.values)) )
    mvp_weights = np.sum(  (sigma_mvp_portfolio**2) * np.linalg.inv(covMat) , axis=1 )

    # weights of the three chosen points  
    m_tild = pd.concat([meanVec, pd.DataFrame({'1s':[1]*len(meanVec.index)}) ], axis=1)
    B = np.transpose(m_tild) @ np.linalg.inv(covMat.values) @ m_tild
    
    _, _, mu_mvp, _, _, _ = q1.sigma_mu(year)
    mu = np.random.uniform(mu_mvp, 1, 3) # mu >= mu_MVP
    mu.sort()
    mu_tild = [[[m], [1]] for m in mu]
    tmp = [np.linalg.inv(covMat.values) @ m_tild @ np.linalg.inv(B) @ i for i in mu_tild]
    rand_3_weights = np.array(tmp[0])
    rand_3_weights = np.append(rand_3_weights, tmp[1], axis=1)
    rand_3_weights = np.append(rand_3_weights, tmp[2], axis=1)
    
    # theoretical market portfolio weights
    R = q1.get_libor_rate(year)
    m_ex = meanVec - R
    mark_port_weights = (1/np.sum(np.linalg.inv(covMat) @ m_ex))*(np.linalg.inv(covMat.values) @ m_ex)
    
    output = np.array(naive_weights)
    output = np.append(output, rand_3_weights, axis=1)
    output = np.append(output, np.transpose(np.array(mvp_weights, ndmin=2)), axis=1)
    
    return np.append(output, mark_port_weights, axis=1)
    


# part b)
def optimizedLassoProblem(data, row, input_tau):
    '''
    Solves the optimization problem to find the optimal
    sparse portfolio weights using Lasso Regression, under
    the constraints that the weights sum to 1 and that the
    linear combination of the weights*mean_of_each_stock sums
    to inputted 'row' value. 

    The regularization hyperparameter for the L1-penality is 'tau'.
    '''
    # helper calculations
    row_1 = row*np.ones([data.shape[0], 1])
    mu_hat = np.mean(data, axis=0)

    w = cvxpy.Variable([data.shape[1], 1])   # our variable
    tau = cvxpy.Parameter(nonneg=True)
    tau.value = input_tau
    # our least squares equation
    objective = cvxpy.Minimize(cvxpy.norm(row_1 - data * w, 2)**2 + tau*cvxpy.norm1(w))
    constraints = [cvxpy.sum(w) == 1, cvxpy.sum(np.transpose(mu_hat) @ w) == row]
    # - tau*cvxpy.norm(w, 1)
    opt_prob = cvxpy.Problem(objective, constraints)
    opt_prob.solve()

    return w.value



if __name__ == '__main__':
    main()