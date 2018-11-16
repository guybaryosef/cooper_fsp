# Pset 3: Question 3
# ECE478: Financial Signal Processing
# Corrolated Brownian Motion
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
    plot_paths(cor_BM_paths, 'Correlated Brownian Motion Pairs')

    # generate and plot 10 pairs of corrolated brownian motion processes
    cor_GBM_proc = simulateGBMprocesses(count, N, delta, cov_mat, alpha_vec, sigma_mat)
    plot_paths(cor_GBM_proc, 'Correlated Brownian Motion Processes Pairs')




# part a)
def generateGaussianRV(mean_vec, covariance_matrix, length):
    '''
    Generate a set of m independent Gaussian random vectors,
    each as a column in a matrix, each with a mean
    perscribed in the mean vector and possessing a covariance
    matrix equal to the inputted one.

    This is done using the Cholesky decomposition.
    '''
    gaussian_mat = np.random.normal(0, 1, [len(mean_vec), length])

    C = np.linalg.cholesky(covariance_matrix)

    for i, mean in enumerate(mean_vec):
        gaussian_mat[i] = mean + C @ gaussian_mat[i]
    return gaussian_mat



# part b)
def generateBrownianMotion(N = 10**5, L = 1000, T = 1):
    '''
    Generate L paths of brownian motion from 
    time 0 to time T, with N steps in between.
    Returns a list of a list of the start and end times
    as well as the steps of the set of paths.

    Note: We are approximating brownian motion using a 
    symmetric random walk.
    '''
    paths = np.zeros([L,N])
    step_increment = 1/np.sqrt(N)
    for path in range(L):
        path_val = np.random.binomial(1, 1/2, N-1)

        for step, step_val in enumerate(path_val):
            if step_val:
                paths[path,step+1] = paths[path,step] + step_increment
            else:
                paths[path,step+1] = paths[path,step] - step_increment
    return [[0, T], paths]



def generateCorrolatedBM(count, N, delta, covariance_mat):
    '''
    Generates 'count' pairs of corrolated brownian motions,
    each with N steps and step size 'delta', as well as 
    each pair possessing the inputted covariance matrix.
    '''

    BM_paths = generateBrownianMotion(N, 2*count, N*delta)
    
    corrolated_BM_paths = []
    for i in range(count):
        ro = covariance_mat[1][0] / np.sqrt(covariance_mat[0][0]*covariance_mat[1][1])

        corrolated_BM_paths.append([BM_paths[1][i], \
                ro*BM_paths[1][i] + np.sqrt(1-ro**2)*BM_paths[1][i+1]])
    return corrolated_BM_paths



def plot_paths(corrolated_paths, title, end_time=1):
    '''
    Plot the inputted paths in a 2 row subplot,
    with the x axis representing time from 0 to 
    the specified end_time.
    '''
    time = np.linspace(0, end_time, corrolated_paths[0][0].shape[0])
    plt.figure()
    plt.suptitle(title);
    for i in range(len(corrolated_paths)):
        plt.subplot(2, len(corrolated_paths)//2, i+1)
        plt.plot(time, corrolated_paths[i][0])
        plt.plot(time, corrolated_paths[i][1])
        plt.title('Instance %d'%(i+1))
    plt.show()



# part c)
def simulateGBMprocesses(count, N, delta, cov_mat, alpha_vec, sigma_mat):
    '''
    '''
    GBM_processes_paths = np.ones([2*count, N])

    for i in range(count):
        for j in range(1, N):
            dWs = generateGaussianRV([0,0], cov_mat, 2)
            GBM_processes_paths[2*i,j] = GBM_processes_paths[2*i,j-1] * \
                                        (1 + alpha_vec[0] + np.sum(sigma_mat[0] @ dWs))

            dWs = generateGaussianRV([0,0], cov_mat, 2)
            GBM_processes_paths[2*i+1,j] = GBM_processes_paths[2*i+1,j-1] * \
                                        (1 + alpha_vec[1] + np.sum(sigma_mat[1] @ dWs))

    output = []
    for i in range(GBM_processes_paths.shape[0]//2):
        output.append([GBM_processes_paths[2*i], GBM_processes_paths[2*i+1]])
    return output



if __name__ == '__main__':
    main()