'''

### Cryptocurrency Trader Agent
### UCB MIDS 2017 Winter Capstone Project
### Ramsey Aweti, Shuang Chan, GuangZhi(Frank) Xie, Jason Xie

### Class: 
###        Environment
### Purpose: 
###        This is utility class used to simulate the cryptocurrency markets.
###        It maintains the state list of the environment.
### Sample Usage:

env = Environment(coin_name="ethereum")
print env.step()
print env.step()
print env.step()
env.plot()

'''

import os
import datetime
import time
import pandas as pd 
import numpy as np
import random
from .utils import *


state_list = ["current_price", "rolling_mean", "rolling_std", "cross_upper_band", "cross_lower_band", "upper_band",
             "lower_band", "price_over_sma"]

class Environment:
    
    # load pricing data
    # initialize the environment variables
    def __init__(self, coin_name="ethereum", states=state_list, num_step = 5000):
        random.seed(time.time())
        self.coin_name = coin_name
        self.states = states
        #dateparse = lambda x: datetime.datetime.fromtimestamp(float(x))
        self.num_step= num_step
        self.all_series = pd.read_csv("%s/cryptocurrencypricehistory/%s_price.csv"
                                  % (os.path.dirname(os.path.abspath(__file__)), self.coin_name), 
                                  parse_dates=["Date"])#,date_parser = dateparse)

        self.all_series.index = self.all_series.sort_values(by=["Date"]).index
        self.all_series = self.all_series.sort_index()

        # self.all_series.set_index('Date', inplace=True)
        # self.all_series['Date'] = self.all_series.index
        # # 设定转换周期period_type  转换为周是'W',月'M',季度线'Q',五分钟'5min',12天'12D'
        # period_type = '5T'
        # # 进行转换，周线的每个变量都等于那一周中最后一个交易日的变量值
        # period_stock_data = self.all_series.resample(period_type, how='last')
        # period_stock_data['Date'] = self.all_series['Date'].resample(period_type, how='first')
        # # 周线的open等于那一周中第一个交易日的open
        # period_stock_data['Open'] = self.all_series['Open'].resample(period_type, how='first')
        # # 周线的high等于那一周中的high的最大值
        # period_stock_data['High'] = self.all_series['High'].resample(period_type, how='max')
        # # 周线的low等于那一周中的low的最大值
        # period_stock_data['Low'] = self.all_series['Low'].resample(period_type, how='min')
        # # 周线的volume等于那一周中volume和money各自的和
        # period_stock_data['Volume'] = self.all_series['Volume'].resample(period_type, how='sum')
        # # 导出数据
        # period_stock_data.to_csv('bitcoin_price_5m.csv', index=False)
        #self.all_series.to_csv("pandas.csv")

        # if self.num_step > 0:
        self.reset()

    # deriving the features used for the state definition
    def __init(self):
        self.isDone = np.zeros(self.series["Open"].shape, dtype=bool)
        self.isDone[-1] = True 

        ### States
        self.rm = self.series["Open"].rolling(window=20, center=False, min_periods=0).mean()
        self.rstd = self.series["Open"].rolling(window=20, center=False, min_periods=0).std()
        self.upper_band, self.lower_band = self.rm + 2 * self.rstd, self.rm - 2 * self.rstd

        ### Mapping states to their names
        self.state_dict = {}
        self.state_dict["current_price"] = self.series["Open"]
        self.state_dict["rolling_mean"] = self.rm
        self.state_dict["rolling_std"] = self.rstd
        self.state_dict["cross_upper_band"] = self.__crossUpperBand()
        self.state_dict["cross_lower_band"] = self.__crossLowerBand()
        self.state_dict["upper_band"] = self.upper_band
        self.state_dict["lower_band"] = self.lower_band
        self.state_dict["price_over_sma"] = self.series["Open"]/self.rm
        
        
    def __crossUpperBand(self):
        crossUpperBand = [0]
        for i in range(1, len(self.series)):
            crossUpperBand.append(self.__checkCrossUpperBand(i)*1)
        return crossUpperBand
    
    
    def __crossLowerBand(self):
        crossLowerBand = [0]
        for i in range(1, len(self.series)):
            crossLowerBand.append(self.__checkCrossLowerBand(i)*1)
        return crossLowerBand
    
        
    def __checkCrossUpperBand(self, curr_index):
        return (
            curr_index - 1 >= 0
            and self.upper_band.loc[curr_index - 1] <= self.state_dict["current_price"][curr_index]
            and self.upper_band.loc[curr_index] > self.state_dict["current_price"][curr_index]
        )
    
    def __checkCrossLowerBand(self, curr_index):
        return (
            curr_index - 1 >= 0
            and self.lower_band.loc[curr_index - 1] >= self.state_dict["current_price"][curr_index]
            and self.lower_band.loc[curr_index] < self.state_dict["current_price"][curr_index]
        )

    ## This is the only place where the state should be exposed
    ''' 
    isDone, state = env.step()
    '''
    # simulate a forward step in the environment, i.e.: moving one day
    def step(self):
        isDone = self.isDone[self.current_index]
        observation = []
        for state in self.states:
            observation.append(self.state_dict[state][self.current_index])
        if not isDone:
            self.current_index += 1
        return isDone, observation

    def getStates(self, states=None):
        if not states:
            states = self.states
        return [self.state_dict[state][self.current_index] for state in states]

    def getStateSpaceSize(self):
        return len(self.states)
    
    
    ## Add method to get current price as it is commonly used
    def getCurrentPrice(self):
        return self.state_dict["current_price"][self.current_index]
    
    def getFinalPrice(self):
        return self.state_dict["current_price"][self.length-1]
    
    def getPriceAt(self, index):
        if index < 0:
            return self.state_dict["current_price"][0]
        if index >= self.length:
            return self.getFinalPrice()
        return self.state_dict["current_price"][index]
    

    def plot(self, states_to_plot=None):
        import matplotlib.pyplot as plt
        if not states_to_plot:
            states_to_plot = self.states

        plt.figure()
        for state in states_to_plot:
            ax = self.state_dict[state].plot()
        ax.legend(states_to_plot)
        plt.show()

    def reset(self):
        self.offset = random.randint(0, self.all_series.shape[0] - self.num_step - 1)
        self.series = self.all_series[self.offset:self.offset + self.num_step]
        self.series.index = [i for i in range(len(self.series))]

        self.length = len(self.series.index)
        self.current_index = 0
        self.__init()
        
    def getReward(self, action):
        a = 0
        if action == Action.BUY:
            a = 1
        elif action == Action.SELL:
            a = -1
            
        price_t = self.getCurrentPrice()
        price_t_minus_1 = self.getPriceAt(self.current_index - 1)
        price_t_minus_n = self.getPriceAt(self.current_index - 10)
        
        r = (1 + a*(price_t - price_t_minus_1)/price_t_minus_1)*price_t_minus_1/price_t_minus_n
        return r
        
        

