# Import packages
import pandas as pd
import numpy as np
import datetime
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ParameterGrid

# Change display options
pd.set_option('display.max_rows', 200)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.precision', 5)
pd.options.display.float_format = '{:.5f}'.format

# Define path
path = '/Users/bobi/Desktop/FIN 427/ML_In_Investing/Gradient_Boosting/Gradient_Boost.py'

# Import single data file
returns01 = pd.read_csv('/Users/bobi/Desktop/FIN 427/ML_In_Investing/Data/Final data 20250312_2300.csv')

# Convert month column to datetime - Assuming "DD-MMM-YYYY" format (e.g., "31-Dec-2022")
returns01['month'] = pd.to_datetime(returns01['month'], format='%d-%b-%Y')

# Split data into training, validation, and test sets
train_cutoff = datetime.datetime(2015, 12, 31)
valid_cutoff = datetime.datetime(2022, 12, 31)
test_cutoff = datetime.datetime(2023, 1, 31)

returns01_train = returns01[returns01['month'] <= train_cutoff]
returns01_valid = returns01[(returns01['month'] > train_cutoff) & (returns01['month'] <= valid_cutoff)]
returns01_test = returns01[(returns01['month'] > valid_cutoff) & (returns01['month'] <= test_cutoff)]

# Filter test set for sector ggroup == 5010
returns01_test_sector_5010 = returns01_test[returns01_test['ggroup'] == 5010]

# Define independent and dependent variables
yd_train = returns01_train['indadjret']
yd_valid = returns01_valid['indadjret']
yd_test_sector_5010 = returns01_test_sector_5010['indadjret']

xd_train = returns01_train[['lag1mcreal',
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
xd_valid = returns01_valid[['lag1mcreal',
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
xd_test_sector_5010 = returns01_test_sector_5010[['lag1mcreal',
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

# Gradient Boosting with Hyperparameter Optimization for Out-of-Sample R²
param_grid_gb = {
    'max_depth': [3, 5],
    'min_weight_fraction_leaf': [0.05, 0.10],
    'n_estimators': [50, 100],
    'max_features': [4, 6],
    'subsample': [0.6, 0.8],
    'learning_rate': [0.01, 0.1]
}

# Initialize variables to store best results
best_r2_valid_gb = -float('inf')
best_params_gb = None

# Convert param_grid to list of dictionaries for iteration
param_list_gb = list(ParameterGrid(param_grid_gb))

# Manual grid search using validation set
print(f"Testing {len(param_list_gb)} hyperparameter combinations for Gradient Boosting...")
for params in param_list_gb:
    gb = GradientBoostingRegressor(random_state=11610, **params)
    gb.fit(xd_train, yd_train)
    r2_valid_gb = gb.score(xd_valid, yd_valid)
    if r2_valid_gb > best_r2_valid_gb:
        best_r2_valid_gb = r2_valid_gb
        best_params_gb = params

print("Best parameters for out-of-sample R² (Gradient Boosting):", best_params_gb)
print("Best validation R² score (Gradient Boosting):", best_r2_valid_gb)

# Train final model with best parameters
gb_optimized = GradientBoostingRegressor(random_state=11610, **best_params_gb)
gbmodel = gb_optimized.fit(xd_train, yd_train)

# Predictions for sector 5010 in January 2023
gbpredictions_test_sector_5010 = gbmodel.predict(xd_test_sector_5010)

# Performance metrics for sector 5010
gbresult_test_sector_5010 = gbmodel.score(xd_test_sector_5010, yd_test_sector_5010)
mae_test_sector_5010 = metrics.mean_absolute_error(yd_test_sector_5010, gbpredictions_test_sector_5010)
mse_test_sector_5010 = metrics.mean_squared_error(yd_test_sector_5010, gbpredictions_test_sector_5010)
rmse_test_sector_5010 = np.sqrt(mse_test_sector_5010)

# Create review DataFrame for sector 5010
gbreview_test_sector_5010 = pd.DataFrame({
    'month': returns01_test_sector_5010['month'],
    'cusip9': returns01_test_sector_5010['cusip9'],
    'comnam': returns01_test_sector_5010['comnam'],
    'permno': returns01_test_sector_5010['permno'],
    'ggroup': returns01_test_sector_5010['ggroup'],
    'indadjret': yd_test_sector_5010,
    'pred_indadjret': gbpredictions_test_sector_5010
})

# Filter for January 2023
jan_2023_sector_5010 = gbreview_test_sector_5010[gbreview_test_sector_5010['month'] == datetime.datetime(2023, 1, 31)]

# Print January 2023 predictions for sector 5010
print("\nGradient Boosting Predictions for January 2023 (Sector 5010):")
print(jan_2023_sector_5010)

# Create performance metrics DataFrame
performance_metrics = pd.DataFrame({
    'Metric': ['R² Score', 'Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error'],
    'Value': [gbresult_test_sector_5010, mae_test_sector_5010, mse_test_sector_5010, rmse_test_sector_5010]
})

# Export results to Excel
with pd.ExcelWriter(path + 'Excel GB Predictions Sector 5010 Jan 2023.xlsx') as writer:
    jan_2023_sector_5010.to_excel(writer, sheet_name='Jan 2023 Predictions', index=False)
    performance_metrics.to_excel(writer, sheet_name='Performance Metrics', index=False)