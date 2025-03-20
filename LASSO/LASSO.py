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

# LASSO regression
lasso_penalty=0.001
lasso = Lasso(alpha=lasso_penalty)

yl_train = returns01_train['indadjret']
xl_train = returns01_train[['lag1mcreal',
                            'zfing01dyadj',
                            'zfing02esg','fing02esgmiss',
                            'zfing03nibadj','fing03nibadjmiss',
                            'zfing04fcfyadj','fing04fcfyadjmiss',
                            'zfing05rdsadj','fing05rdsadjmiss',
                            'zfing06_invpegadj','fing06_invpegadjmiss',
                            'zfing07epadj','fing07epadjmiss',
                            'zfing08sadadj','fing08sadadjmiss',
                            'zfing09shoadj','fing09shoadjmiss',
                            'zfing10shiadj','fing10shiadjmiss',
                            'zfing11ret5adj','fing11ret5adjmiss',
                            'zfing12empadj','fing12empadjmiss',
                            'zfing13sueadj','fing13sueadjmiss',
                            'zfing14erevadj','fing14erevadjmiss']]

yl_valid = returns01_valid['indadjret']
xl_valid = returns01_valid[['lag1mcreal',
                            'zfing01dyadj',
                            'zfing02esg','fing02esgmiss',
                            'zfing03nibadj','fing03nibadjmiss',
                            'zfing04fcfyadj','fing04fcfyadjmiss',
                            'zfing05rdsadj','fing05rdsadjmiss',
                            'zfing06_invpegadj','fing06_invpegadjmiss',
                            'zfing07epadj','fing07epadjmiss',
                            'zfing08sadadj','fing08sadadjmiss',
                            'zfing09shoadj','fing09shoadjmiss',
                            'zfing10shiadj','fing10shiadjmiss',
                            'zfing11ret5adj','fing11ret5adjmiss',
                            'zfing12empadj','fing12empadjmiss',
                            'zfing13sueadj','fing13sueadjmiss',
                            'zfing14erevadj','fing14erevadjmiss']]

lasso.fit(xl_train, yl_train)
lasso_coef_train = lasso.fit(xl_train, yl_train).coef_
lasso_score_train = lasso.score(xl_train, yl_train)
print(lasso.intercept_)
print(lasso_coef_train)
print(lasso_score_train)
print(xl_train.columns)
print(pd.Series(lasso_coef_train, index=xl_train.columns))
y_pred_lasso_train = lasso.predict(xl_train)
ssr_lasso_train = np.sum((yl_train - y_pred_lasso_train)**2)
sstl_train      = np.sum((yl_train - np.mean(yl_train))**2)
rsq_lasso_train = 1 - (ssr_lasso_train/sstl_train)
rmse_lasso_train = np.sqrt(np.mean((yl_train - y_pred_lasso_train)**2))

print(f'Sum of squared difference between y values and predicted y values (SSR): {ssr_lasso_train:.5f}')
print(f'Sum of squared difference between y values and average y values (SST): {sstl_train:.5f}')
print(f'R-squared = 1 - SSR/SST: {rsq_lasso_train:.5f}')
print(f'Square root of the mean squared error: {rmse_lasso_train:.5f}')

y_pred_lasso_valid = lasso.predict(xl_valid)
ssr_lasso_valid = np.sum((yl_valid - y_pred_lasso_valid)**2)
sst_lasso_valid = np.sum((yl_valid - np.mean(yl_valid))**2)
rsq_lasso_valid = 1 - (ssr_lasso_valid/sst_lasso_valid)
rmse_lasso_valid = np.sqrt(np.mean((yl_valid - y_pred_lasso_valid)**2))
print(f'Sum of squared difference between y values and predicted y values in test sample (SSR): {ssr_lasso_valid:.5f}')
print(f'Sum of squared difference between y values and average y values in test sample (SST): {sst_lasso_valid:.5f}')
print(f'R-squared in test sample = 1 - SSR/SST: {rsq_lasso_valid:.5f}')
print(f'Square root of the mean squared error in test sample: {rmse_lasso_valid:.5f}')

# LASSO regression on validation sample
lasso_valid = Lasso(alpha=lasso_penalty)
lasso_valid.fit(xl_valid, yl_valid)
lasso_coef_valid = lasso_valid.fit(xl_valid, yl_valid).coef_
lasso_score_valid = lasso_valid.score(xl_valid, yl_valid)
print(lasso_valid.intercept_)
print(lasso_coef_valid)
print(lasso_score_valid)
print(xl_valid.columns)
print(pd.Series(lasso_coef_valid, index=xl_valid.columns))
y_pred_lasso_valid2 = lasso_valid.predict(xl_valid)
ssr_lasso_valid2 = np.sum((yl_valid - y_pred_lasso_valid2)**2)
sstl_valid2      = np.sum((yl_valid - np.mean(yl_valid))**2)
rsq_lasso_valid2 = 1 - (ssr_lasso_valid2/sstl_valid2)
rmse_lasso_valid2 = np.sqrt(np.mean((yl_valid - y_pred_lasso_valid2)**2))

print(f'Sum of squared difference between y values and predicted y values (SSR): {ssr_lasso_valid2:.5f}')
print(f'Sum of squared difference between y values and average y values (SST): {sstl_valid2:.5f}')
print(f'R-squared = 1 - SSR/SST: {rsq_lasso_valid2:.5f}')
print(f'Square root of the mean squared error: {rmse_lasso_valid2:.5f}')

# Create a dataframe of coefficients
lasso_coef_df = pd.DataFrame({'Feature': ['intercept']       + xl_train.columns.tolist(),
'Coefficients from training set'  : [lasso.intercept_]       + lasso.coef_.tolist(),
'Coefficients from validation set': [lasso_valid.intercept_] + lasso_valid.coef_.tolist()})
print(lasso_coef_df)

# Create a Numpy array of R-squared and then a Dataframe, for export to Excel
exportarray01_lasso = np.array([[rsq_lasso_train, rsq_lasso_valid, rsq_lasso_valid2]])
exportdf01_lasso = pd.DataFrame(exportarray01_lasso,columns=['rsq_lasso_train', 'rsq_lasso_valid', 'rsq_lasso_valid2'])
print(exportdf01_lasso)

# Export to Excel
with pd.ExcelWriter(path + 'Excel 06 OLS and LASSO 20250212_2056.xlsx') as writer:
    lasso_coef_df.to_excel(writer, sheet_name='coef_lasso')
    exportdf01_lasso.to_excel(writer, sheet_name='exportdf01_lasso')