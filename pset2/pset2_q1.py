#
# Pset 2: Question 1
# ECE478: Financial Signal Processing
# Binomail Asset Pricing Model
# By: Guy Bar Yosef
#

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt


def main():

    s0 = 1 # price of stock at time 0
    u = 1.005
    r = 0.003
    d = 1.002
    N = 100   # path length
    L = 1000  # number of simulations
    p = 1/2
    probabilities = [0.4, 0.6] # real probabilities

    p_tilda, _ = risk_free_prob(u, d, r) # the risk-free probabilities 

    protfolio_strategy_approach(p, s0, N, u, d, r)

    derivative_vals = simulate_derivative_values(probabilities, p_tilda, L, N, s0, u, d)

    v0 = v0_calculation(s0, p_tilda, L, r, N, u, d)
    
    summarize(derivative_vals, v0, r, N)



# Part a)
exp_Rn = lambda u,d,n,p, : np.log10(u/d)*(n*p) + n*np.log10(d)
var_Rn = lambda u,d,n,p : (np.log(u/d)**2) * n*p*(1-p)


# Part b)
def risk_free_prob(u,d,r):
    '''
    Given a u, d, and r, returns the 
    risk-free probabilities p and q.
    '''
    p_tilda = (1+r-d)/(u-d)
    q_tilda = (u-1-r)/(u-d)
    return p_tilda, q_tilda


# Part c)
def stock_history(s0, path, u, d):
    s = [s0] # list of stock price over time 
    for i in path:
        if i == 1:
            s.append(s[-1]*u)
        else:
            s.append(s[-1]*d)
    return s


def replicate_portfolio(Vn_func, p, s0, path, N, u, d, r):
    '''
    Given a derivate price at time n and the
    path that led to it, return the replicating
    portfolio and the portfolio strategy over the path.
    '''
    p_tilda, q_tilda = risk_free_prob(u, d, r)
    stock_price = stock_history(s0, path, u, d)
    K = strike_price(exp_Rn(u,d,N,p), s0)
    delta = []
    
    for i in range(len(path)-1, 0, -1):
        vn_h = Vn_func(K, s0, path[:i] + [1], u, d)
        vn_l = Vn_func(K, s0, path[:i] + [0], u, d)

        cur_delta_val = (vn_h - vn_l) / (stock_price[i]*u - stock_price[i]*d)
        delta.insert(0, cur_delta_val)

    x0 = (1/(1+r))*(p_tilda*Vn_func(K, s0, [1], u, d) + q_tilda*Vn_func(K, s0, [0], u, d) )
    delta.insert(0, (Vn_func(K, s0, [1], u, d) - Vn_func(K, s0, [0], u, d))/(s0*u - s0*d) )
    return x0, delta


# Part d)
class Vn:
    @staticmethod
    def _find_stock_price(path, s0, u, d):
        s = s0
        for i in path:
            if i == 1:
                s *= u
            else:
                s *= d
        return s

    @staticmethod
    def european_call(K, s0, path, u, d):
        '''
        Finds the payout of a european call option given
        a path, strike price K, initial stock price s0, and
        u and d values.
        '''
        stock_price = Vn._find_stock_price(path, s0, u, d)

        if K >= stock_price: 
            return 0
        else:
            return stock_price - K

    @staticmethod
    def european_put(K, s0, path, u, d):
        '''
        Finds the payout of a european put option given
        a path, strike price K, initial stock price s0, and
        u and d values.
        '''
        stock_price = Vn._find_stock_price(path, s0, u, d)

        if K <= stock_price: 
            return 0
        else:
            return K - stock_price

    @staticmethod
    def max_stock_price(K, s0, path, u, d):
        '''
        Finds the maximum price a stock reaches given 
        a path, initial stock price s0, and
        u and d values.
        note: the K is simply to keep the format of the other derivatives.
        '''
        max_val = s0
        cur_val = s0
        for i in path:
            if i == 1:
                cur_val *= u
            else:
                cur_val *= d
            if cur_val > max_val:
                max_val = cur_val
        return max_val 


# Part e)
strike_price = lambda exp_Rn, s0 : s0 * np.exp(exp_Rn)

def simulate_derivative_values(probabilities, p_tilda, L, N, s0, u, d):
    '''
    Derives the expected values of the 3 derivatives after N steps using
    a monte carlo approach (with L being the number of generated paths).
    '''
    expected_value_derivative = []

    for p in probabilities:
        outer_iter = []
        for p2 in probabilities + [p_tilda]:
            inner_iter = []
            for _ in range(L):
                cur_path = np.random.binomial(1, p, size=[N]) 

                K = strike_price(exp_Rn(u,d,N,p2), s0)
                tmp = [ Vn.max_stock_price(K, s0, cur_path, u, d),
                          Vn.european_call(K, s0, cur_path, u, d),
                           Vn.european_put(K, s0, cur_path, u, d) ]

                inner_iter.append(tmp)
            outer_iter.append( np.mean(inner_iter, axis=0) )
        expected_value_derivative.append(outer_iter)
    return expected_value_derivative


# Part f)
def v0_calculation(s0, p_tilda, L, r, N, u, d):
    '''
    Using the martingale property, returns that 
    risk-neutral price of derivative(s) valued after N steps.
    '''
    v0_values = []
    for _ in range(L):
        cur_path = np.random.binomial(1, p_tilda, N) 

        # assuming the probability for K is also p_tilda
        K = strike_price(exp_Rn(u,d,N,p_tilda), s0)
        Vn_max =  Vn.max_stock_price(K, s0, cur_path, u, d)
        Vn_e_call = Vn.european_call(K, s0, cur_path, u, d)
        Vn_e_put =   Vn.european_put(K, s0, cur_path, u, d)

        v0_values.append([ Vn_max, Vn_e_call, Vn_e_put])
    v0_values = np.mean(v0_values, axis=0)
    return v0_values/((1+r)**N)



# Part g)
def protfolio_strategy_approach(p, s0, N, u, d, r):
    '''
    Using the portfolio strategy approach, this graphs
    the portfolio strategies over time for each of the 
    derivatives, for 3 different random paths.
    '''
    stock_history_path = []
    time = range(0, N, 1)
    print('\n')
    for i in range(3):
        cur_path = np.random.binomial(1, p, N)

        stock_history_path.append(stock_history(s0, cur_path, u, d))

        x0_max, delta_max = replicate_portfolio(Vn.max_stock_price, p, s0, cur_path, N, u, d, r)
        x0_cal, delta_cal = replicate_portfolio(  Vn.european_call, p, s0, cur_path, N, u, d, r)
        x0_put, delta_put = replicate_portfolio(   Vn.european_put, p, s0, cur_path, N, u, d, r)

        print('Path:', i+1, 'Arbitrage-free Price of Derivative')
        print('Max Stock:', x0_max, end='--')
        print('European Call', x0_cal, end='--')
        print('European Put', x0_put)

        plt.figure()
        plt.plot(time, delta_max, label='Max Stock Price')
        plt.plot(time, delta_cal, label='European Call')
        plt.plot(time, delta_put, label='European Put')
        plt.title('Portfolio strategy for path %s'% str(i+1))
        plt.xlabel('Steps')
        plt.ylabel('Portfolio Strategy')
        plt.legend()
        plt.show()

    print('\n')

    plt.figure()
    for i, stock in enumerate(stock_history_path):
        plt.plot(time, stock[:N], label='Stock $S_%s$' % i)
    plt.legend()
    plt.title('Stock Price of the Three Paths')
    plt.xlabel('Steps')
    plt.ylabel('Stock Price')
    plt.show()

# Part h)
def summarize(derivative_vals, v0, r, N):
    '''
    Prints a summary of the expected discounted prices of the derivaites.
    Inputs:
        deriavtive_vals: a 2-d list, with the rows representing a proability
            and columns a derivative.
        v0: a list whose columns represent derivaties (needs to match columns
            of 'derivative_vals').
        r: discounted risk-free interest rate.
        N: Length of path.
    '''
    decimal_places = 4
    E1 = [ np.round(i/((1+r)**N), decimal_places) for i in derivative_vals[0] ]
    E2 = [ np.round(i/((1+r)**N), decimal_places) for i in derivative_vals[1] ]
    v0 = np.round(v0, decimal_places)
    table = Table({'Derivative':['Max Stock Price', 'European Call', 'European Put'],
                            'P=0.4, K_=0.4': E1[0], 'P=0.6, K_p=0.4': E2[0], 
                            'P=0.4, K_=0.6': E1[1], 'P=0.6, K_p=0.6': E2[1],
                            'P=0.4, K_=0p_tilday': E1[2], 'P=0.6, K_p=p_tilda': E2[2],
                            '      V0      ': v0 })

    print(table)



if __name__ == '__main__':
    main()