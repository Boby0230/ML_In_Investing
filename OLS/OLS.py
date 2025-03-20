# Import packages
# You need to install the package scikit-learn in order for lines referring to sklearn to run
# You need to install the package python-docx in order for lines referring to docx to run
import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.linear_model import Lasso
import datetime
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn import tree
from sklearn.tree import _tree
from docx import Document

# # Change the number of rows and columns to display
pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.5f}'.format

# Define a path for import and export
path = '/Users/bobi/Desktop/FIN 427/Final Paper/OLS'

# Import and view data
returns01 = pd.read_csv('/Users/bobi/Desktop/FIN 427/Final Paper/Data/Final data 20250312_2300.csv')
returns01['month'] = pd.to_datetime(returns01['month'], format='%d-%b-%Y')
print(returns01.dtypes)
print(returns01.head(10))
print(returns01.columns)

datecut1 = datetime.datetime(2015, 12, 31)
datecut2 = datetime.datetime(2022, 12, 31)
print(datecut1)
print(datecut2)
returns01_train   = returns01[returns01['month'] <= datecut1]
returns01_valid   = returns01[(returns01['month'] >  datecut1)   & (returns01['month'] < datecut2)]
returns01_all     = returns01[returns01['month'] < datecut2]
returns01_predict = returns01[returns01['month'] == datecut2]
print(returns01_train.shape)
print(returns01_valid.shape)
print(returns01_all.shape)
print(returns01_predict.shape)

y_train = returns01_train['indadjret']
x_train = returns01_train[['lag1mcreal',
                            'fing01dyadj','fing01dyadjmiss',
                            'fing02esg','fing02esgmiss',
                            'fing03nibadj','fing03nibadjmiss',
                            'fing04fcfyadj','fing04fcfyadjmiss',
                            'fing05rdsadj','fing05rdsadjmiss',
                            'fing06_invpegadj','fing06_invpegadjmiss',
                            'fing07epadj','fing07epadjmiss',
                            'fing08sadadj','fing08sadadjmiss',
                            'fing09shoadj','fing09shoadjmiss',
                            'fing10shiadj','fing10shiadjmiss',
                            'fing11ret5adj','fing11ret5adjmiss',
                            'fing12empadj','fing12empadjmiss',
                            'fing13sueadj','fing13sueadjmiss',
                            'fing14erevadj','fing14erevadjmiss']]

y_valid = returns01_valid['indadjret']
x_valid = returns01_valid[['lag1mcreal',
                            'fing01dyadj','fing01dyadjmiss',
                            'fing02esg','fing02esgmiss',
                            'fing03nibadj','fing03nibadjmiss',
                            'fing04fcfyadj','fing04fcfyadjmiss',
                            'fing05rdsadj','fing05rdsadjmiss',
                            'fing06_invpegadj','fing06_invpegadjmiss',
                            'fing07epadj','fing07epadjmiss',
                            'fing08sadadj','fing08sadadjmiss',
                            'fing09shoadj','fing09shoadjmiss',
                            'fing10shiadj','fing10shiadjmiss',
                            'fing11ret5adj','fing11ret5adjmiss',
                            'fing12empadj','fing12empadjmiss',
                            'fing13sueadj','fing13sueadjmiss',
                            'fing14erevadj','fing14erevadjmiss']]

print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)
print(x_train.columns)
print(x_valid.columns)

# OLS regression with out-of-sample prediction
x_train = sm.add_constant(x_train)
x_valid = sm.add_constant(x_valid)
model = sm.OLS(y_train, x_train).fit()
print_model = model.summary()
ols_coef = model.params
ols_rsq = model.rsquared

y_pred_train = model.predict(x_train)
ssr_train = np.sum((y_train - y_pred_train)**2)
sst_train = np.sum((y_train - np.mean(y_train))**2)
rsq_train = 1 - (ssr_train/sst_train)
rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2))

y_pred_valid = model.predict(x_valid)
ssr_valid    = np.sum((y_valid - y_pred_valid)**2)
sst_valid    = np.sum((y_valid - np.mean(y_valid))**2)
rsq_valid    = 1 - (ssr_valid/sst_valid)
rmse_valid   = np.sqrt(np.mean((y_valid - y_pred_valid)**2))

print(print_model)
print(ols_coef)

print(f'R-squared in training sample from rsquared function: {model.rsquared:.5f}')
print(f'Sum of squared difference between y values and predicted y values in training sample (SSR): {ssr_train:.5f}')
print(f'Sum of squared difference between y values and average y values in training sample (SST): {sst_train:.5f}')
print(f'R-squared in training sample = 1 - SSR/SST: {rsq_train:.5f}')
print(f'Square root of the mean squared error in training sample: {rmse_train:.5f}')

print(f'Sum of squared difference between y values and predicted y values in validation sample (SSR): {ssr_valid:.5f}')
print(f'Sum of squared difference between y values and average y values in validation sample (SST): {sst_valid:.5f}')
print(f'R-squared in validation sample = 1 - SSR/SST: {rsq_valid:.5f}')
print(f'Square root of the mean squared error in validation sample: {rmse_valid:.5f}')

# Regression on validation sample
model2 = sm.OLS(y_valid, x_valid).fit()
print_model2 = model2.summary()
ols_coef2 = model2.params
ols_rsq2 = model2.rsquared

print(print_model2)
print(ols_coef2)

y_pred_valid2 = model2.predict(x_valid)
ssr_valid2    = np.sum((y_valid - y_pred_valid2)**2)
sst_valid2    = np.sum((y_valid - np.mean(y_valid))**2)
rsq_valid2    = 1 - (ssr_valid2/sst_valid2)
rmse_valid2   = np.sqrt(np.mean((y_valid - y_pred_valid2)**2))

print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_valid2:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_valid2:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_valid2:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_valid2:.5f}')

# Create a Numpy array of R-squared and then a Dataframe, for export to Excel
exportarray01 = np.array([[rsq_train, rsq_valid, rsq_valid2]])
exportdf01 = pd.DataFrame(exportarray01,columns=['rsq_train', 'rsq_valid', 'rsq_valid2'])
print(exportdf01)

