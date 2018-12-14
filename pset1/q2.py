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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():

    # get data for the user-specified year
    year = input('Input year to test between 2000 and 2016: ')
    
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
    tau = np.linspace(0, 5000, 20)
    row = np.linspace(-5,5,20)
    data = pd.read_csv('./by_years/48_IP_eq_w_daily_returns_'+str(year)+'.csv').values[:,1:]
    
    optimizedLassoProblem(data, row, tau, year)



# part a)
def sparcifyWeights(weights, p):
    '''
    Given a set of weights (may be 2-d) and an order p,
    this function returns the sparsified portfolios under
    the p-norm, as well as the optimal number of non-zero
    coefficients (weights). 
    '''
    sparce_weights = copy.deepcopy(weights)
    k_val = []
    for i in range(weights.shape[1]):
        min_val = 0.1*np.linalg.norm(weights[:,i], ord=p)
        best_val = 48
        best_weights = weights[:,i]

        # k is the number of non-zero coefficients
        for k in range(1, weights[:,i].size):
            g_k = copy.deepcopy(weights[:,i])
            count_to_zero = weights[:,i].size - k
            
            while count_to_zero > 0:
                if np.sum(abs(g_k)) == 0:
                    count_to_zero = 0
                    continue
                to_zero = np.where( abs(g_k) == np.min(abs(g_k)[np.nonzero(abs(g_k))]))[0]
                if count_to_zero < to_zero.size:
                    to_zero = to_zero[:count_to_zero]
                for j in to_zero:
                    g_k[j] = 0
                count_to_zero -= to_zero.size
            
            if np.sum(g_k) == 0:
                continue

            g_k /= np.sum(g_k)
            
            # test for the sparsification condition
            g_k_norm = np.linalg.norm(weights[:,i] - g_k, ord=p)
            if g_k_norm < min_val:
                min_val = g_k_norm
                best_val = k
                best_weights = g_k
            
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
def optimizedLassoProblem(data, ro_vals, input_tau_vals, year):
    '''
    Solves the optimization problem to find the optimal
    sparse portfolio weights using Lasso Regression, under
    the constraints that the weights sum to 1 and that the
    linear combination of the weights*mean_of_each_stock sums
    to inputted 'row' value. 

    This function loops over the optimization problem with the 
    inputted values of ro and tau and plots the results (this 
    is akin to grid searching, only instead of searching for 
    values we keep track of them and then plot our results.

    We plot a rho-sigma graph (return-risk graph) and a scatter
    plot of the sparce portfolio S(w) for each rho and tau.
    '''
    _, covMat = q1.find_mean_cov(year)

    tau_subset = [input_tau_vals[0], input_tau_vals[input_tau_vals.size//2], input_tau_vals[-1]]

    # plot sigma vs. ro points, for fixed tau
    plt.figure()

    for input_tau in tau_subset:
        sigma_vals = []

        for ro in ro_vals:
            # helper calculations
            ro_1 = ro*np.ones([data.shape[0], 1])
            mu_hat = np.mean(data, axis=0)

            w = cvxpy.Variable([data.shape[1], 1])   # our variable
            tau = cvxpy.Parameter(nonneg=True)
            tau.value = input_tau

            # our least squares equation
            objective = cvxpy.Minimize(cvxpy.norm(ro_1 - data * w, 2)**2 + tau*cvxpy.norm1(w))
            constraints = [cvxpy.sum(w) == 1, cvxpy.sum(np.transpose(mu_hat) @ w) == ro]
            # - tau*cvxpy.norm(w, 1)
            opt_prob = cvxpy.Problem(objective, constraints)
            opt_prob.solve()
        
            sigma_vals.append(np.sum(np.sqrt( np.transpose(w.value) @ covMat.values @ w.value)))

        plt.plot(sigma_vals, ro_vals, label='Tau = '+str(input_tau))

    plt.xlabel('$\sigma$')
    plt.ylabel('$\\rho$')
    plt.title('The Optimization functions Return-Risk plot with fixed Tau')
    plt.legend()
    plt.savefig('./outputs/FSP_pset1_q2_rho_sigma_plot.pdf')

    points_p = []
    points_t = []
    points_n = []
    for ro in ro_vals:
        for input_tau in input_tau_vals:
            # helper calculations
            ro_1 = ro*np.ones([data.shape[0], 1])
            mu_hat = np.mean(data, axis=0)

            w = cvxpy.Variable([data.shape[1], 1])   # our variable
            tau = cvxpy.Parameter(nonneg=True)
            tau.value = input_tau
            # our least squares equation
            objective = cvxpy.Minimize(cvxpy.norm(ro_1 - data * w, 2)**2 + tau*cvxpy.norm1(w))
            constraints = [cvxpy.sum(w) == 1, cvxpy.sum(np.transpose(mu_hat) @ w) == ro]
            # - tau*cvxpy.norm(w, 1)
            opt_prob = cvxpy.Problem(objective, constraints)
            opt_prob.solve()

            points_p.append(ro)
            points_t.append(input_tau)
            points_n.append(np.linalg.norm(sparcifyWeights(w.value, 1)[0][:,0], ord=1))
        
    # plotting the norm of the sparce weights versus tau and ro
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_t, points_p, points_n)
    ax.set_xlabel('$\\tau$')
    ax.set_ylabel('$\\rho$')
    ax.set_zlabel('Sparce Portfolio')
    plt.title('The Sparce Portfolios')
    plt.savefig('./outputs/FSP_pset1_q2_scatter_plot.pdf')



if __name__ == '__main__':
    main()