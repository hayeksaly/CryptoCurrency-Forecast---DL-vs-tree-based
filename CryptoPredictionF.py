''' Import libraries'''
import pandas_datareader as web # to read data from yahoo.finance
import datetime as dt # to handle datetime values
import pandas as pd # for dataframe manipulation
import numpy as np # manipulate arrays and calculations
import matplotlib.pyplot as plt # plot models results
import xgboost as xgb # machine learning model
from sklearn.ensemble import RandomForestRegressor # machine learning model
from sklearn.preprocessing import MinMaxScaler # to normalize data
from sklearn.metrics import mean_squared_error # compute error result of the models
import warnings # erase warnings
import tensorflow as tf # deep learning library
import inquirer # used to manipulate user input
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # manipulate time series data (divide it between train and test)
from statsmodels.graphics.tsaplots import plot_acf # plot correlation
from tensorflow.keras.models import Sequential # handle layers of deep learning
from tensorflow.keras.layers import LSTM, Dense # deep learning model
warnings.filterwarnings("ignore") # ignore warnings in output


''' User choice of coin '''
questions = [
  inquirer.List('crypto',
                message="Chose a cryptocurrency",
                choices=['BTC', 'ETH', 'XRP'],
            ),
]  # prepare the questions that the user will chose from
answers = inquirer.prompt(questions) # inquire the user to chose one from coins
print(f'You entered {answers["crypto"]}') # display the user answer


''' Data Gathering '''

coin  = answers["crypto"] # the coin chosen by user
currency = 'USD' # currency used to measure coin
# real time data preparation
start = dt.datetime(2013, 1, 1) # set start date of data gathering
end = dt.datetime.now() # set end date as the date of the usage of code to ensure real time data preprocessing
coin = web.DataReader(f'{coin}-{currency}', "yahoo", start, end) # read the data from yahoo.finance
print(coin)

''' Data Preprocessing '''

coin.reset_index(inplace=True) #reset data index
coin['Date']=pd.to_datetime(coin['Date']) # convert date in dataframe to datetime values
coin['Date']= coin['Date'].dt.strftime('%Y-%m-%d') # Set the date fromat as Year-Month-Day
coin = coin.sort_values(by = 'Date') # sort rows of dataframe by date
print('Data scraped:\n') # print the obtained dataframe
print(coin)


''' Autocorrelation of last 60 days with current price '''

data = coin[['Date', 'Close']].set_index(['Date']) # chose data to be correlated : Date vs close price
plot_acf(data, lags=60) # plot the correlation of last 60 days (lags = 60) vs predicted date
# plt.show() # show correlation

''' LSTM MODEL '''

# prepare data to enter the model
close_data = coin['Close'].values # taking close price from the dataframe
close_data = close_data.reshape((-1,1)) # reshape the data to enter the model

#split data into train and test
split_percent = 0.80 # split percentage = 80% : training on 80% of data testing on 20%
split = int(split_percent*len(close_data))

# train and test data: train is 80% of data(historical data) and test 20%
close_train = close_data[:split] # train prices on historical values
close_test = close_data[split:] # test prices on most recent values

date_train = coin['Date'][:split] # train date on historical data
date_test = coin['Date'][split:] # test on most recent data

# chose dates to enter the model: 60 days
look_back = 15

# prepare the data for model: last 60 days for train and next day predict
train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20) # prepare data train for last 60 days
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1) # prepare test data 1 day prediction

# LSTM model
model = Sequential()
model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
) # add 10 layers in the model to input
model.add(Dense(1)) # output one layer the prediction for one day price
model.compile(optimizer='adam', loss='mse')

num_epochs = 25
model.fit_generator(train_generator, epochs=num_epochs, verbose=1) # fit the model on train data

# predict test values
prediction = model.predict_generator(test_generator)

# reshape data
close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))

# plot prediction results and real values
plt.figure(figsize=(14,5))
plt.plot(date_test[-len(prediction):], close_test[-len(prediction):], color = 'red', label = 'Real coin Price')
plt.plot(date_test[-len(prediction):], prediction, color = 'green', label = 'Predicted coin Price by LSTM model')
plt.title('coin Price Prediction using LSTM')
plt.show()

# group results in dataframe
LSTM_res = pd.DataFrame()
LSTM_res['LSTM date'] = date_test[-len(prediction):]
LSTM_res['real values'] = close_test[-len(prediction):]
LSTM_res['Prediction LSTM values'] = prediction[-len(prediction):]
print('LSTM results: \n')
print(LSTM_res)


''' 5 DAYS prediction'''
close_data = close_data.reshape((-1))


def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]

    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back - 1:]

    return prediction_list


def predict_dates(num_prediction):
    last_date = date_test.values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
    return prediction_dates


num_prediction = 6
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)
print(forecast)
print(forecast_dates)
forecasted = pd.DataFrame()
forecasted['Date'] = forecast_dates[1:]
forecasted['Coin Price by LSTM'] = forecast[1:]


''' Prepare data for XGBOOST and Random Forest'''

# split train and test data
data_training = coin[coin['Date']< '2021-10-01'].copy() # train data is data before 01-10-2021
data_test = coin[coin['Date']>= '2021-10-01'].copy() # test data after 01-10-2021
training_data = data_training.drop(['Date', 'Volume', 'Adj Close'], axis = 1) # drop date, volume and adj close columns: Don't contribute in predictions

# prepare train data for Random forest and XGboost models
scaled_data = training_data['Close'].values.reshape(-1, 1)
prediction_days = 60
x_train, y_train = [], []
for x in range (prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0]) # append the price values as features to train the model
    y_train.append(scaled_data[x, 0]) # append the price target
x_train, y_train = np.array(x_train), np.array(y_train) # transform train and test data to array

# Random forest regressor model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(x_train, y_train) # fit train data in the random forest model prepared

# XGboost regressor model
xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1, max_depth=1, alpha=10,
                          n_estimators=200) # xgboost model preparation
xg_reg.fit(x_train, y_train) # fit train data in XGBoost model

# prepare test data
data_testt = data_test.drop(['Date', 'Volume', 'Adj Close'], axis = 1)
scaler = MinMaxScaler(feature_range=(0,1)) #normalize data
scaled_dataT = scaler.fit_transform(data_testt['Close'].values.reshape(-1, 1))
scaled_data = data_testt['Close'].values.reshape(-1, 1) # reshape test data

prediction_days = 60 # 60 days contribute in the prediction
# prepare test data for evaluation (prediction)
x_test, y_test = [], []
for x in range (prediction_days, len(scaled_data)):
    x_test.append(scaled_data[x-prediction_days:x, 0]) # 60 days values append to the features
    y_test.append(scaled_data[x, 0]) # next day price (target price)

x_test, y_test = np.array(x_test), np.array(y_test) # convert test data to array

''' Compute Results '''

# predict results of test data
preds = xg_reg.predict(x_test)

# compute RMSE for XGboost
rmse = np.sqrt(mean_squared_error(y_test, preds))
pred_tomorrow = xg_reg.predict([x_test[-1]]) # predict tomorrow price by inputing last 60 days coin prices
print('XGboost prediction for tomorrow price:\n')
print(pred_tomorrow)
# print('XGboost Mean squared error')
# print("RMSE: %f" % (rmse))



# Compute RMSE for Random forest prediction
res = rf_model.predict(x_test) # predict test data results
rmse = np.sqrt(mean_squared_error(y_test, res)) # compute rmse
pred_tomorrow_rf = rf_model.predict([x_test[-1]]) # predict tomorrow price by inputing last 60 days coin prices
print('Random forest prediction coin tomorrow price:\n')
print(pred_tomorrow_rf)
# print('Random forest Mean squarred error:\n')
# print("RMSE: %f" % (rmse))

date_t = [] 
for i in range(0, len(y_test)):
    date_t.append(data_test['Date'][-len(y_test):].values[i])
#create dataframe that holds the results: Predicted vs Real values
results = pd.DataFrame()
results['Real values'] = y_test
results['Random forest predictions'] = res
results['XGboost predictions'] = preds
results['Date'] = date_t

print('Results prediction dataframe: \n', results)

# plot random forest predictions
plt.figure(figsize=(14,5))
plt.plot(date_t, y_test, color = 'red', label = 'Real coin Price')
plt.plot(res, color = 'green', label = 'Predicted coin Price by Random Forest model')
plt.title('coin Price Prediction using Random Forest')

# plot XGboost predictions
plt.figure(figsize=(14,5))
plt.plot(date_t, y_test, color = 'red', label = 'Real coin Price')
plt.plot(preds, color = 'green', label = 'Predicted coin Price by XGbosst model')
plt.title('coin Price Prediction using XGbosst')
# plt.show()

''' 5 DAYS prediction for random Forest and XGboost'''

def predict(num_prediction, model):
    prediction_list = x_test[-1]

    for _ in range(num_prediction):
        x = prediction_list[-60:]
        out = model.predict([x])
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[60 - 1:]

    return prediction_list


def predict_dates(num_prediction):
    last_date = date_test.values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction + 1).tolist()
    return prediction_dates



num_prediction = 6
price_pred = predict(num_prediction, rf_model)
date_pred = predict_dates(num_prediction)
price_pred_xg = predict(num_prediction, xg_reg)
print(price_pred)
print(date_pred)
forecasted['Random Forest model price prediction'] = price_pred[1:]
forecasted['XGBOOST model price prediction'] = price_pred_xg[1:]

forecasted.to_csv('Forecasted_5DayRange.csv')
