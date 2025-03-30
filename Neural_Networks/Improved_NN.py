import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import datetime
from numpy.random import seed
from tensorflow.random import set_seed
from keras.models import Sequential
from keras.layers import Input, Dense, BatchNormalization
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
TF_ENABLE_ONEDNN_OPTS = 0

# Change the number of rows and columns to display
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.precision', 5)
# pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Define a path for import and export
path = '/Users/bobi/Desktop/FIN 427/ML_In_Investing/Neural_Networks/'

# Import and view data
returns01 = pd.read_csv('/Users/bobi/Desktop/FIN 427/ML_In_Investing/Data/Final data 20250312_2300.csv')
returns01['month'] = pd.to_datetime(returns01['month'], format='%d-%b-%Y')

datecut1 = datetime.datetime(2016, 1, 31)
datecut2 = datetime.datetime(2023, 1, 31)

returns01_train = returns01[returns01['month'] <= datecut1]
returns01_valid = returns01[(returns01['month'] > datecut1) & (returns01['month'] < datecut2)]
returns01_all = returns01[returns01['month'] < datecut2]
returns01_predict = returns01[returns01['month'] == datecut2]

y_train = returns01_train['indadjret']
x_train = returns01_train[['zlnlag1mcreal', 'finlag1bmmiss', 'zfinlag1bm', 'zfing01dyadj', 'fing02esgmiss',
                            'zfing02esg', 'fing03nibadjmiss', 'zfing03nibadj', 'fing04fcfyadjmiss', 'zfing04fcfyadj',
                            'fing05rdsadjmiss', 'zfing05rdsadj', 'fing06_invpegadjmiss', 'zfing06_invpegadj',
                            'fing07epadjmiss', 'zfing07epadj', 'fing08sadadjmiss', 'zfing08sadadj', 'fing09shoadjmiss',
                            'zfing09shoadj', 'fing10shiadjmiss', 'zfing10shiadj', 'fing11ret5adjmiss', 'zfing11ret5adj',
                            'fing12empadjmiss', 'zfing12empadj', 'fing13sueadjmiss', 'zfing13sueadj', 'fing14erevadjmiss',
                            'zfing14erevadj']]

y_valid = returns01_valid['indadjret']
x_valid = returns01_valid[['zlnlag1mcreal', 'finlag1bmmiss', 'zfinlag1bm', 'zfing01dyadj', 'fing02esgmiss',
                           'zfing02esg', 'fing03nibadjmiss', 'zfing03nibadj', 'fing04fcfyadjmiss', 'zfing04fcfyadj',
                           'fing05rdsadjmiss', 'zfing05rdsadj', 'fing06_invpegadjmiss', 'zfing06_invpegadj',
                           'fing07epadjmiss', 'zfing07epadj', 'fing08sadadjmiss', 'zfing08sadadj', 'fing09shoadjmiss',
                           'zfing09shoadj', 'fing10shiadjmiss', 'zfing10shiadj', 'fing11ret5adjmiss', 'zfing11ret5adj',
                           'fing12empadjmiss', 'zfing12empadj', 'fing13sueadjmiss', 'zfing13sueadj', 'fing14erevadjmiss',
                           'zfing14erevadj']]

y_all = returns01_all['indadjret']
x_all = returns01_all[['zlnlag1mcreal', 'finlag1bmmiss', 'zfinlag1bm', 'zfing01dyadj', 'fing02esgmiss',
                       'zfing02esg', 'fing03nibadjmiss', 'zfing03nibadj', 'fing04fcfyadjmiss', 'zfing04fcfyadj',
                       'fing05rdsadjmiss', 'zfing05rdsadj', 'fing06_invpegadjmiss', 'zfing06_invpegadj',
                       'fing07epadjmiss', 'zfing07epadj', 'fing08sadadjmiss', 'zfing08sadadj', 'fing09shoadjmiss',
                       'zfing09shoadj', 'fing10shiadjmiss', 'zfing10shiadj', 'fing11ret5adjmiss', 'zfing11ret5adj',
                       'fing12empadjmiss', 'zfing12empadj', 'fing13sueadjmiss', 'zfing13sueadj', 'fing14erevadjmiss',
                       'zfing14erevadj']]

y_predict = returns01_predict['indadjret']
x_predict = returns01_predict[['zlnlag1mcreal', 'finlag1bmmiss', 'zfinlag1bm', 'zfing01dyadj', 'fing02esgmiss',
                               'zfing02esg', 'fing03nibadjmiss', 'zfing03nibadj', 'fing04fcfyadjmiss', 'zfing04fcfyadj',
                               'fing05rdsadjmiss', 'zfing05rdsadj', 'fing06_invpegadjmiss', 'zfing06_invpegadj',
                               'fing07epadjmiss', 'zfing07epadj', 'fing08sadadjmiss', 'zfing08sadadj', 'fing09shoadjmiss',
                               'zfing09shoadj', 'fing10shiadjmiss', 'zfing10shiadj', 'fing11ret5adjmiss', 'zfing11ret5adj',
                               'fing12empadjmiss', 'zfing12empadj', 'fing13sueadjmiss', 'zfing13sueadj', 'fing14erevadjmiss',
                               'zfing14erevadj']]

# Initialize the random seed for the neural network. Nets select random weights to start the process.
seed(24754)
set_seed(11610)

# Start the neural network with more layers and early stopping
nnet0 = Sequential()

# Input layer
nnet0.add(Input(shape=(x_train.shape[1],)))

# Hidden layers with BatchNormalization and increased neurons
nnet0.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
nnet0.add(BatchNormalization())
nnet0.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
nnet0.add(BatchNormalization())

# Output layer
nnet0.add(Dense(1, activation='linear'))

# Compile the model
nnet0.compile(optimizer='adam', loss='mse')

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.00001)

# Reduce Learning Rate on Plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Fit the model
history0 = nnet0.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_valid, y_valid),
                     callbacks=[early_stopping, reduce_lr])

# Calculate R-squared scores for training and validation sets
preds_train = nnet0.predict(x_train)
print('R-squared train:', r2_score(y_train, preds_train))
preds_valid = nnet0.predict(x_valid)
print('R-squared valid:', r2_score(y_valid, preds_valid))

# Now training on the full dataset to make predictions for January 2023
nnet1 = Sequential()

nnet1.add(Input(shape=(x_all.shape[1],)))
nnet1.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
nnet1.add(BatchNormalization())
nnet1.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))
nnet1.add(BatchNormalization())
nnet1.add(Dense(1, activation='linear'))

nnet1.compile(optimizer='adam', loss='mse')

# Train on all data with early stopping and learning rate reduction
history1 = nnet1.fit(x_all, y_all, epochs=100, batch_size=32, validation_data=(x_valid, y_valid),
                     callbacks=[early_stopping, reduce_lr])

# Calculate R-squared scores
preds_all = nnet1.predict(x_all)
print('R-squared all:', r2_score(y_all, preds_all))

# Make predictions for the validation and test data
preds_predict = nnet1.predict(x_predict)

