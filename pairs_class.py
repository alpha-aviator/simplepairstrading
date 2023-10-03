import os
import random
import numpy as np
import yfinance as yf
import torch.nn as nn
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from matplotlib import pyplot as plt
from itertools import combinations
from scipy.stats import norm


class PT(nn.Module):
        
    # default attributes -> use class methods to change "default" attributes
    annual_basis = 252
    riskfree_rate = float(0.03)

    def __init__(self, ticker_list, start_date, end_date):
        super(PT, self).__init__()
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date
        self.coint_pairs = []
        self.pair_zscore = dict()
    

    def __repr__(self):  # code rep of user inputs for check
        return f"input_data = (Tickers = {self.ticker_list}, Start Date = {self.start_date}, End Date = {self.end_date})"


    def get_xldata(self,filepath):
        
        self.data = pd.read_excel(filepath,sheet_name="prices",index_col=0)
        self.data = self.data.loc[self.start_date:self.end_date,self.ticker_list]
        self.data_close = self.data
        self.data_adj_close = self.data_close


    def get_yfdata(self,data_list):
        self.data = yf.download(data_list, start=self.start_date, end=self.end_date)
        print("===== DOWNLOAD COMPLETE =====")
        self.data_adj_close = self.data["Adj Close"]
        self.data_low = self.data["Low"]
        self.data_open = self.data["Open"]
        self.data_close = self.data["Close"]
        self.data_high = self.data["High"]
        self.data_volume = self.data["Volume"]

        return pd.DataFrame(self.data)
        
    def stationarity_test(self, x, threshold):
        """
        input:
        x: a list of scalar values
        threshold: significance level
        output: print out message on stationarity
        """
        pvalue = adfuller(x)[1]
        if pvalue < threshold:
            return 'p-value is ' + str(pvalue) + '. The series is likely stationary.'
        else:
            return 'p-value is ' + str(pvalue) + '. The series is likely non-stationary.'
        
    def cointegration_test(self,data,pair):
        # build linear regression model
        # Extract prices for two stocks of interest
        # target var: Y; predictor: X
        Y = data[pair[0]]
        X = data[pair[1]]

        # estimate linear regression coefficients of stock1 on stock2
        X_with_constant = sm.add_constant(X)
        model = OLS(Y, X_with_constant).fit()
        residuals = Y - model.predict()
        print(model.params)
        return residuals
        
        # # alternative approach
        # residuals2 = Y - (model.params['const'] + model.params[stocks[1]] * X)
        # # check if both residuals are the same
        # print(residuals.equals(residuals2))
        # # test residuals for stationarity
        
    def stationarity_test_residual(self, residuals, threshold:float):
        adf_test = adfuller(residuals)
        print(f"ADF test statistic: {adf_test[0]}")
        print(f"p-value: {adf_test[1]}")

        if adf_test[1] < threshold:
            print("The two stocks are cointegrated.")
            return True
        else:
            print("The two stocks are not cointegrated.")
            return False 
        
    def get_pairs(self,ticker_list):
        # get all pairs of stocks
        self.pairs_list = list(combinations(ticker_list, 2))
        return self.pairs_list
    
    def get_spread(self,pair):
        # calculate the spread for GOOG and MSFT
        # Y = pair[1]
        # X = pair[0]
        Y = self.data_close[pair[1]]
        X = self.data_close[pair[0]]
        print(pair, "pair printed")
        # estimate linear regression coefficients
        X_with_constant = sm.add_constant(X)
        model = OLS(Y, X_with_constant).fit()
        # obtain the spread as the residuals
        spread = Y - model.predict()
        spread.plot(figsize=(12,6))
        print(spread)
        print(type(spread))
        return spread
    
    def get_z_score(self,spread,z_crit):
        
        # illustrate z score by generating a standard normal distribution with mu 0 and sd 1
        
        # # input: unbounded scalar, assumed to be in the range of [-5,-5] in this case
        # x = np.linspace(-5, 5, 100)
        # # output: probability between 0 and 1
        # y = norm.pdf(x, loc=0, scale=1)
        # # set up the plot
        # fig, ax = plt.subplots()
        # # plot the pdf of normal distribution
        # ax.plot(x, y)
        # # shade the area corresponding to a z-score of >=1.96 and <=-1.96

        # x_shade = np.linspace(z_crit, 5, 100)
        # y_shade = norm.pdf(x_shade, loc=0, scale=1)
        # ax.fill_between(x_shade, y_shade, color='red', alpha=0.3)
    
        # x_shade2 = np.linspace(-5, -z_crit, 100)
        # y_shade2 = norm.pdf(x_shade2, loc=0, scale=1)
        # ax.fill_between(x_shade2, y_shade2, color='red', alpha=0.3)
        # # add labels and a title
        # ax.set_xlabel('Z-score')
        # ax.set_ylabel('Probability density')
        # # add a vertical line to indicate the z-score of 1.96 and -1.96
        # ax.axvline(x=z_crit, linestyle='--', color='red')
        # ax.axvline(x=-z_crit, linestyle='--', color='red')
        # # display the plot
        # plt.show()    
        
        # convert to z score
        # z-score is a measure of how many standard deviations the spread is from its mean
        # derive mean and sd using a moving window
        window_size = 10
        spread_mean = spread.rolling(window=window_size).mean()
        spread_std = spread.rolling(window=window_size).std()
        zscore = (spread - spread_mean) / spread_std
        # zscore.plot(figsize=(12,6))
        # plt.show()
        # remove initial days with NA
        # first_valid_idx = zscore.first_valid_index()
        # zscore = zscore[first_valid_idx:]
        # zscore.rename("z_score")
        print(zscore)
        return zscore
    
    def get_buyandhold_ticker(self,tickers):
        data = self.data_adj_close[tickers]
        for x in tickers:
            data[str(x+"_log_return")] = np.log(data[x] / data[x].shift(1))
        self.buyandhold = data
        return None

# use dict to initialise config
setup = {"ticker_list":['GOOG','MSFT','AAPL','META','NFLX','AMD','MU','AMZN','INTC','CRM','TSM', 'NVDA', 'CSCO', 'ADBE', 'ORCL','TXN'], 
         "start_date":"2020-01-01",
         "end_date":"2023-10-01"}


pt1 = PT(setup["ticker_list"], start_date = setup["start_date"], end_date = setup["end_date"]) # create instance

pt1.get_yfdata(pt1.ticker_list) # get data
pt1.get_pairs(pt1.ticker_list)
pt1.get_buyandhold_ticker(tickers=pt1.ticker_list)

# run coint test, get pairs, form df, run test, return coint pairs list
for x in pt1.pairs_list:
    df = pt1.data_adj_close.loc[:,x[0:2]] # create df for prices in the pair to test
    test_resd = pt1.cointegration_test(df, list(df.columns)) # coint test for pair, get residuals
    if pt1.stationarity_test_residual(test_resd,threshold=0.01) == True: # test residuals and append to coint pairs list
        pt1.coint_pairs.append(x)

# run spread and z score for pair in coint pair
for x in pt1.coint_pairs:
    # makes (pair) a key, then runs spread n zscore and stores z score series as value
    spread = pt1.get_spread(x)
    df = pd.DataFrame(pt1.get_z_score(spread,z_crit=1.96))
    # rename zscore column
    df.columns = ["z_score"]
    df["spread"] = spread
    
    # append logbuyandhold of tickers to df
    for y in range(len(x)):
        a = x[y]
        a = str(a + "_log_return")
        df[a] = pt1.buyandhold[a]
        
    pt1.pair_zscore[x] = df

    # print(df)
    #append the two tickers and close to df -> do you really need to do it?
    pt1.pair_zscore[x][x[0]], pt1.pair_zscore[x][x[1]] = pt1.data_close[x[0]], pt1.data_close[x[1]]
    
print("========DATA SCREENING IS COMPLETE========")    

# signals and returns
for x in pt1.pair_zscore:
    
    backtest = pt1.pair_zscore[x].dropna()
    backtest[str(x[0] + "_position")] = np.where(backtest["z_score"] > 2, -1, 0)
    backtest[str(x[1] + "_position")] = np.where(backtest["z_score"] > 2, 1, 0)
    backtest[str(x[0] + "_position")] = np.where(backtest["z_score"] < -2, 1, 0)
    backtest[str(x[1] + "_position")] = np.where(backtest["z_score"] < -2, -1, 0)
    backtest[str(x[0] + "_position")] = np.where((-1 < backtest["z_score"]) & (backtest["z_score"] < 1), 0, 0)
    backtest[str(x[1] + "_position")] = np.where((-1 < backtest["z_score"]) & (backtest["z_score"] < 1), 0, 0)
    
    # calculate returns
  
    backtest[str(x[0] + "_returns")] = np.where(backtest[str(x[0] + "_position")] == 1, backtest[str(x[0]+"_log_return")] * backtest[str(x[0] + "_position")], 0)
    backtest[str(x[1] + "_returns")] = np.where(backtest[str(x[1] + "_position")] == 1, backtest[str(x[1]+"_log_return")] * backtest[str(x[1] + "_position")], 0)
    total_returns = backtest[str(x[0] + "_returns")] + backtest[str(x[1] + "_returns")]
    cumulative_returns = (1 + total_returns).cumprod()
    print(cumulative_returns)
    
    





