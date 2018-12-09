# ECE:478 Financial Signal Processing
# Pset 4: Financial Time-Series Analysis
# By: Guy Bar Yosef
#
# Preprocessing for the pset

import pandas as pd

def preprocess(): 
    '''
    Read in data, clean it up and seperate it into seperate years of exactly
    250 days (truncates as needed).

    The data is:
        - Farma and French 48 benchmark dataset (will be our portfolio)
        - S&P 500 (will be our market portfolio... initially, at least)
    '''
    # Farma and French 48 benchmark dataset
    daily_data = pd.read_csv("./input/48_Industry_Portfolios_daily.CSV", low_memory=False)
    daily_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    eq_weight_daily_data = daily_data.loc[24267:]

    eq_weight_daily_data.set_index('Date', inplace=True) # set the date as the index
    daily_return_year = split_into_years(eq_weight_daily_data) # list of our new datasets 
    
    for i, data in enumerate(daily_return_year):
        
        data[:250].to_csv('./by_years/48_IP_eq_w_daily_returns_' + str(i+2000) + '.csv')

    # S&P 500 
    sp_daily = pd.read_csv("./input/dailySP500.CSV")

    temp_sp_ID =  [ i.replace('-', '') for i in sp_daily['Date'] ]
    temp_sp_returns = 100 * (sp_daily['Adj Close'] - sp_daily['Open'] ) / sp_daily['Open']

    sp_returns = pd.DataFrame({'Date':  temp_sp_ID, 'Return': temp_sp_returns })
    sp_returns.set_index('Date', inplace=True)

    sp_returns_year = split_into_years(sp_returns)

    for i, data in enumerate(sp_returns_year):
        data[:250].to_csv('./by_years/SP_daily_returns_' + str(i+2000) + '.csv')



def split_into_years(data, beg_year = 2000, amount_years = 17):
    '''
    Seperate each of the datasets into different files for
    each of the years specified (begining at beg year and for amount_years 
    amount of years).

    Note: year doesn't have to start on Jan 1st and end on Dec 31st, 
    due to the possibility of these dates landing on weekends.
    '''
    new_datasets = []
    for i in range(amount_years):

        # due to weekends, need to make sure we start with correct index
        start_date = (beg_year+i)*(10**4) + 101
        while (str(start_date) not in data.index):
            start_date += 1

        end_date = (beg_year+i)*(10**4) + 1231
        while (str(end_date) not in data.index):
            end_date -= 1

        new_datasets.append(data.loc[str(start_date):str(end_date)])
    return new_datasets