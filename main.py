# FE520 - Final project
#
# Authors(Group 25): Yoohan Ko, Nick Ordonez, Kshitij Rai
# "I pledge my honor I have abided by the Stevens Honor System"

from linreg import *

# testing the functionality of the linear regression package

tickers = ['ACE', 'ABT', 'ANF', 'ACN', 'ADBE', 'AMD', 'AES', 'AET',
           'AFL', 'A', 'GAS', 'APD', 'ARG', 'AKAM', 'AA', 'ALXN', 'ATI', 'AGN', 'ALL', 'ANR']
start = '2016-01-01'
end = '2018-01-01'

# first, we want to create an instance of our linear regression package
lr = LinReg(tickers)

# pulling prices from Yahoo Finance for the date range specified
lr_prices = lr.getPrices(start, end)
print("Printing the first few rows of the prices dataframe: ")
print(lr_prices.head)

# setting a window of 30 days
lr_window = lr.setWindow(30)
print(lr_window[0])

# calculating variance and covariance
lr_var = lr.getVariance()
print(str(lr_var[0]))
lr_cov = lr.getCovariance()
print(str(lr_cov[0][0]))

# calculating the slopes (Beta) of the dataset
lr_slopes = lr.getSlopes()
print("Calculating Slopes (Beta): \n" + str(lr_slopes[0]))

# calculating Intercepts from the slopes of the dataset
lr_intercepts = lr.getIntercepts()
print("Finding Intercepts: \n" + str(lr_intercepts[0]))

# plotting the linear regression
lr.plotWindow(150, "AA")

# get the R squared value
lr_rsq = lr.getRsquared(200, ["AA", "AKAM"])
print(lr_rsq)

# get the standard error coefficient
lr_err = lr.getStdcoeff(200, ["AA", "AKAM"])
print(lr_err)

# get the T Scores
lr_tsc = lr.getTscores(200, ["AA", "AKAM"])
print(lr_tsc)

# get the P Values
lr_pval = lr.getPvalues(["AA", "AKAM"])
print(lr_pval)

