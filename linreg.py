# FE520 - Final project
# Description: This is a linear regression package that was created for the final project of FE 520, Python
# for financial applications.

# Authors(Group 25): Yoohan Ko, Nick Ordonez, Kshitij Rai
# "I pledge my honor I have abided by the Stevens Honor System"

import pandas as pd
import math
import numpy as np
import scipy.stats as stat
import yfinance as yhf
import matplotlib.pyplot as plt


# This class pulls data from yfinance
class LinReg:
    def __init__(self, tickers):
        # convert from list of stock tickers to char
        tickers = ', '.join(tickers)
        self.tickers = tickers
        self.SP500 = '^GSPC'

        print("Creating Linear regression model instance created using the following tickers: ")
        print(tickers)

        # other instance variables to be used for calculations
        # getPrices
        self.tickers_frame = None
        self.close_prices = None
        self.sp500_frame = None
        self.close_sp500 = None
        self.all_prices = None

        # setWindow
        self.window = None
        self.all_prices_samples = None

        # getVariance
        self.window_means = None
        self.window_variance = None
        self.variance_func = None

        # getCovariance
        self.window_covariance = None

        # getSlopes
        self.window_slopes = None

        # getIntercepts
        self.window_intercept = None

        # getRsquared
        self.Rsq = None
        self.Rsq_series = None
        self.resi_squa = None
        self.R_squared = None

        # getStdcoeff
        self.stderr_series = None

        # getTscores
        self.t_scores = None

        # getPvalues
        self.p_val = None

    # pull the prices of the stocks for the specified dates from Yahoo Finance API
    def getPrices(self, startdate, enddate):
        print("==============================================================================")
        print("Pulling prices from Yahoo Finance API - yhf")
        print("Using the dates: " + str(startdate) + str(enddate))
        self.tickers_frame = pd.DataFrame(yhf.Tickers(self.tickers).history(start=startdate, end=enddate))
        self.close_prices = self.tickers_frame['Close'].dropna(axis=1)
        self.sp500_frame = pd.DataFrame(yhf.Ticker(self.SP500).history(start=startdate, end=enddate))
        self.close_sp500 = self.sp500_frame['Close']
        self.all_prices = pd.concat([self.close_prices, self.close_sp500], axis=1)
        # print("Printing the first few lines of prices downloaded from yahoo finance")
        # print(self.all_prices.head)
        return self.all_prices

    # pull the prices of the stocks for the specified dates & window
    def setWindow(self, window):
        # Sample from the dataset using specified moving window
        # Create a list of subsets of the closing prices for the tickers and S&P 500
        print("==============================================================================")
        print("Setting a window of: " + str(window) + " days. ")
        self.window = window
        self.all_prices_samples = []
        for i in range(window, len(self.close_sp500)):
            self.all_prices_samples.append(self.all_prices[(i - window):i])
        print("Dataset sampled from a window size of: " + str(window))
        return self.all_prices_samples

    def getVariance(self):
        # Calculate the mean values for each stock in each window
        # Store these means as pandas series objects
        print("==============================================================================")
        print("Calculating Variance...")
        self.window_means = []
        for i in range(0, len(self.all_prices_samples)):
            self.window_means.append(self.all_prices_samples[i].apply(np.mean))

        # Calculate the variance of S&P 500 closing prices for each window
        self.window_variance = []
        self.variance_func = lambda x: (x - self.window_means[i]['Close']) ** 2
        for i in range(0, len(self.all_prices_samples)):
            mean_x = self.window_means[i]['Close']
            var = sum(self.all_prices_samples[i]['Close'].map(self.variance_func)) / (
                        len(self.all_prices_samples[i]['Close']) - 1)
            self.window_variance.append(var)
        return self.window_variance

    def getCovariance(self):
        # Calculate the covariances of each stock and the S&P 500 within each window
        print("==============================================================================")
        print("Calculating Covariance...")
        cov_x = lambda x: (x - self.window_means[i]['Close'])
        cov_y = lambda y: (y - self.window_means[i][j])
        self.window_covariance = []
        for i in range(0, len(self.all_prices_samples)):
            cov_list = []
            index_list = []
            for j in self.all_prices_samples[i].drop(labels='Close', axis=1).columns:
                covariance = sum(self.all_prices_samples[i][j].map(cov_y) *
                                 self.all_prices_samples[i]['Close'].map(cov_x)) / \
                             (len(self.all_prices_samples[i]['Close']) - 1)
                cov_list.append(covariance)
                index_list.append(j)
            self.window_covariance.append(pd.Series(cov_list, index=index_list))
        return self.window_covariance

    def getSlopes(self):
        # Get the slope of the regression lines for each stock in each window
        # The slope is the same as the beta value
        print("==============================================================================")
        print("Calculating Slopes...")
        self.window_slopes = []
        for i in range(0, len(self.all_prices_samples)):
            self.window_slopes.append(self.window_covariance[i] / self.window_variance[i])
        # print(self.window_slopes)
        return self.window_slopes

    def getIntercepts(self):
        # Get the intercept values for each stock in each window
        print("==============================================================================")
        print("Calculating Intercepts...")
        self.window_intercept = []
        for i in range(0, len(self.all_prices_samples)):
            inter_list = []
            index_list = []
            w1 = self.window_slopes[i]
            mean_close = np.mean(self.all_prices_samples[i]['Close'])
            for j in self.all_prices_samples[i].drop(labels='Close', axis=1).columns:
                w0 = np.mean(self.all_prices_samples[i][j]) - w1[j] * mean_close
                inter_list.append(w0)
                index_list.append(j)
            self.window_intercept.append(pd.Series(inter_list, index=index_list))
        print(self.window_intercept[0])
        return self.window_intercept

    def plotWindow(self, windownum, tick):
        # Demonstrating that each set of points was divided into 30 day windows
        # and each window has a matching line function for each stock
        print("==============================================================================")
        print("Plotting window: " + str(windownum) + " of stock: "+ str(tick))
        plt.scatter(self.all_prices_samples[windownum]['Close'], self.all_prices_samples[windownum][tick], c='black')
        x = list(self.all_prices_samples[windownum]['Close'])
        y = []
        for i in x:
            y1 = i * (self.window_slopes[windownum][tick]) + self.window_intercept[windownum][tick]
            y.append(y1)
        plt.plot(x, y, c='red')


    def getRsquared(self, windownum, list_tickers):
        # Calculate the R-squared value for a specific stock in a given window
        print("==============================================================================")
        print("Calculating the R-Squared value for a window of " + str(windownum) + str(list_tickers))
        self.Rsq = []
        indexes = []
        for i in list_tickers:
            list_tickers = i
            tot_squa = sum((self.all_prices_samples[windownum][list_tickers] - self.window_means[windownum][
                list_tickers]) ** 2)
            x = list(self.all_prices_samples[windownum]['Close'])
            y = []
            for i in x:
                y1 = i * (self.window_slopes[windownum][list_tickers]) + self.window_intercept[windownum][list_tickers]
                y.append(y1)

            resi_squa = sum((self.all_prices_samples[windownum][list_tickers] - y) ** 2)
            R_squared = 1 - (resi_squa / tot_squa)
            self.Rsq.append(R_squared)
            indexes.append(list_tickers)
        self.Rsq_series = pd.Series(self.Rsq, index=indexes)
        return self.Rsq_series

    def getStdcoeff(self, windownum, list_tickers):
        # Get the standard error coefficients
        print("==============================================================================")
        print("Calculating the Standard Error Coefficients for a window of " + str(windownum) + str(list_tickers))
        serror = []
        indexes = []
        for i in list_tickers:
            list_tickers = i
            n = len(self.all_prices_samples[windownum])
            se_coef = (math.sqrt(sum((self.all_prices_samples[windownum][list_tickers] -
                                      self.all_prices_samples[windownum]['Close']) ** 2) / (n - 2)) / math.sqrt(
                sum((self.all_prices_samples[windownum]['Close'] - self.window_means[windownum]['Close']) ** 2)))
            serror.append(se_coef)
            indexes.append(list_tickers)
        self.stderr_series = pd.Series(serror, index=indexes)
        return self.stderr_series

    def getTscores(self, windownum, list_tickers):
        # Get the t-scores
        print("==============================================================================")
        print("Calculating the T score for a window of " + str(windownum) + str(list_tickers))
        tsc = []
        indexes = []
        for i in list_tickers:
            list_tickers = i
            t = self.window_slopes[windownum][list_tickers] / self.stderr_series[list_tickers]
            tsc.append(t)
            indexes.append(list_tickers)
        self.t_scores = pd.Series(tsc, index=indexes)
        return self.t_scores

    def getPvalues(self, list_tickers):
        # Get p-values using scipy stats package
        print("==============================================================================")
        print("Calculating the P values for tickers: " + str(list_tickers))
        df = len(self.all_prices_samples) - 2
        self.p_val = 1 - stat.t.cdf(self.t_scores[list_tickers], df=df)
        temp_df = pd.Series(self.p_val, index=list_tickers)
        return temp_df
