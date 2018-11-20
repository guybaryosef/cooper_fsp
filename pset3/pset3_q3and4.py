# Pset 3: Questions 3 and 4
# ECE478: Financial Signal Processing
# Corrolated Brownian Motion and Portfolio Volatility
# By: Guy Bar Yosef

import numpy as np
import matplotlib.pyplot as plt


def main():
    # user defined variables
    N = 250
    count = 10
    cov_mat = np.array([[1, 0.75], [0.75, 0.9]])
    alpha_vec = [0.1/N, 0.1/N]
    sigma_mat = np.array([alpha_vec, alpha_vec])
    delta = 0.01
    
    # generate and plot 10 pairs of corrolated brownian motion pairs
    cor_BM_paths = generateCorrolatedBM(count, N, delta, cov_mat)
    plot_paths(cor_BM_paths, 'Correlated Brownian Motion Pairs', N*delta)

    # generate and plot 10 pairs of corrolated brownian motion processes
    cor_GBM_proc = simulateGBMprocesses(count, N, delta, cov_mat, alpha_vec, sigma_mat)
    plot_paths(cor_GBM_proc, 'Correlated Brownian Motion Processes Pairs', N*delta)

    # portfolio simulations
    w_min = np.transpose([[0, 1]])
    w_max = np.transpose([[1505540, 2509324]]) # this seems off to say the least...
    port_paths = portfolioSimulations(count, N, delta, w_min, w_max, cov_mat, sigma_mat, alpha_vec)
    plot_paths(port_paths, 'Miniimally and Maximally Volatile Portfolios')



# part 3a)
def generateGaussianRV(mean_vec, covariance_matrix, length):
    '''
    Generate a set of m independent Gaussian random vectors,
    each as a column in a matrix, each with a mean
    perscribed in the mean vector and possessing a covariance
    matrix equal to the inputted one.

    This is done using the Cholesky decomposition.
    '''
    gaussian_mat = np.random.normal(0, 1, [length, len(mean_vec)])

    C = np.linalg.cholesky(covariance_matrix)

    for i, mean in enumerate(mean_vec):
        gaussian_mat[:,i] = mean + C @ gaussian_mat[:,i]
    return gaussian_mat



# part 3b)
def generateCorrolatedBM(count, N, delta, covariance_mat):
    '''
    Generates 'count' pairs of corrolated brownian motions,
    each with N steps and step size 'delta', as well as 
    each pair possessing the inputted covariance matrix.
    '''    
    corrolated_BM_paths = np.array(count*[np.ones([2,N])])

    for i in range(count):
        variances = generateGaussianRV([0]*(N-1), covariance_mat, 2)

        for j in range(1,N):
            corrolated_BM_paths[i,0,j] = corrolated_BM_paths[i,0,j-1] + variances[0,j-1]
            corrolated_BM_paths[i,1,j] = corrolated_BM_paths[i,1,j-1] + variances[1,j-1]

    return corrolated_BM_paths



def plot_paths(corrolated_paths, title, end_time=1):
    '''
    Plot the inputted paths in a 2 row subplot,
    with the x axis representing time from 0 to 
    the specified end_time.
    '''
    time = np.linspace(0, end_time, corrolated_paths[0][0].shape[0])
    plt.figure()
    plt.suptitle(title)
    for i in range(len(corrolated_paths)):
        plt.subplot(2, len(corrolated_paths)//2, i+1)
        plt.plot(time, corrolated_paths[i][0])
        plt.plot(time, corrolated_paths[i][1])
        plt.title('Instance %d'%(i+1))
        plt.xlabel('time')
    plt.savefig('./figures/Pset3_q3_%s'%(title))
    plt.show()



# part 3c)
def simulateGBMprocesses(count, N, delta, cov_mat, alpha_vec, sigma_mat):
    '''
    Simulates pairs of stochastic processes that are each driven by correlated
    brownian motions with a 'cov_mat' covariance matrix.
    Will return 'count' pairs of of processes, each with N steps.
    '''
    GBM_processes_paths = np.array(count*[np.ones([2, N])])

    for i in range(count):
        for j in range(1, N):
            dWs = generateGaussianRV([0,0], cov_mat, 2)
            GBM_processes_paths[i,0,j] = GBM_processes_paths[i,0,j-1] * \
                                        (1 + alpha_vec[0] + np.sum(sigma_mat[0] @ dWs))

            dWs = generateGaussianRV([0,0], cov_mat, 2)
            GBM_processes_paths[i,1,j] = GBM_processes_paths[i,1,j-1] * \
                                        (1 + alpha_vec[1] + np.sum(sigma_mat[1] @ dWs))
    return GBM_processes_paths



# 4)
def portfolioSimulations(pairs, N, delta, w_min, w_max, cov_mat, sigma, alpha):
    '''
    Given the weights for the portfolios of two stocks driven by correlated
    brownian motions that correspond to the least and most volatile portfolios
    (w_min and w_max respectivly), this function simulates 'pairs'instances
    of the portfolios and returns their paths.
    '''
    processes_paths = simulateGBMprocesses(pairs, N, delta, cov_mat, alpha, sigma)
    
    paths = []
    for i in range(pairs):
        V_min_paths = np.ones([N])
        V_max_paths = np.ones([N])

        for j in range(1,N):
            V_min_paths[j] = np.sum(np.transpose(w_min) @ processes_paths[i,:,j])
            V_min_paths[j] = np.sum(np.transpose(w_max) @ processes_paths[i,:,j])
        
        paths.append([V_min_paths, V_max_paths])

    return paths



if __name__ == '__main__':
    main()