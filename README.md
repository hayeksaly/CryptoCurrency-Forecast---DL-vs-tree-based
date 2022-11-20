# CryptoCurrency-Forecast---DL-vs-tree-based
Integrate and use machine learning models, compare the results to study the trends and patterns of historical data, and predict the price of a chosen coin.

BACKGROUND AND PROBLEM STATEMENT:
Predicting a cryptocurrency in the stock market has always been a challenge. And this is not an easy task as many factors can influence the price prediction. The price of coins is highly volatile, which makes the prediction more challenging. The most data we have on this type of predictions is the historical price.
The goal of this project is to integrate and use machine learning models, compare the results to study the trends and patterns of historical data, and predict the price of a chosen coin.
 

Approach taken:  SUPERVISED:
In the subject of trying to predict stock market, like all the projects related to data and predicting; 3 approach can usually be taken: Supervised, unsupervised and Reinforcement learning; depending on the complexity of the project; availability of data along with abundance and type of the data we have. Unsupervised is usually used for unlabeled data; where the result is unknown; it is used to separate data and cluster it based on groups; Reinforcement is usually used to let the algorithm change with changement of outcome; it excels traditional supervised and unsupervised learning and it is used for more complex projects.
The project that has been done use supervised learning algorithm. The historical price of data is known and abundant. Will be using the historical data to feed to the algorithm so it can learn patterns from and base its predictions on.
Methodology:
The first approach used to acquire the goal of the project was to base the prediction on historical data; since historical data is well known, and it has the most higher effect on the current price of the coin; and since the project is generalized on many coins. This can be developed in more specific projects later and is subject to improvement. 
3 non-linear models used. Two tree based algorithms and one using deep learning algorithms. These algorithms are then evaluated to choose the best model.





Libraries used:
Pandas_datareader : to read data from yahoo.finance
datetime : to handle datetime values
Pandas : dataframe manipulation
Numpy: manipulate arrays and calculations
Matplotlib: plot models results
xgboost  : machine learning model
sklearn: handle machine learning models 
tensorflow  : deep learning library
inquirer : used to manipulate user input
DATA GATHERING:
Data is gathered by webscraping coins prices from yahoo.finance website through a library implemented in python called datareader that reads coins historical and actual prices directly from the website. We enter the coin desired along with the currency ($ in our case), and the beginning date with end date desired. In our project we needed as much historical data to be able to predict on, so we chose the date from 1-1-2013 until now; which means the date of lunching of program. The until now function lets the program be real time: Each time we run it it gives us the data until this time. 

Features – Target Correlation:
 
The feature- Target correlation graph shows the correlation of the last 60 days price effect on the current price. The previous day price shows highest correlation with percentage = 0.99%, it decreases respectively to attain around 77% positive correlation on day 60 which proves the high correlation of the previous 60 days price on the current crypto price.  
USER CHOICE OF COIN 
The program is designed to let the user chose the coin that he wishes to predict: This is possible through giving the user a list of coins he can chose from. And this choice and user interface is provided by inquirer library which lets the user chose from the coins: The program holder can change the list of coins he wishes to appear for the user. After choice of coin, the coin chosen by the user is put as input to the datareader library, along with date of training data and currency ($). The output is a dataframe containing the date, along with open and close price of the coin for the dates cited.
 

DATA PREPARATION AND PREPROCESSING
Data is prepared by dividing it into target and features: 60 days for model implementation and the next day as the target. Data preparation is done on 2 phases: One preparation for tree based algorithms (XGBOOST and Random Forest), and one for the deep neural networks algorithm: LSTM.

Training and Evaluation of results of the 3 DIFFERENT ALGORITHMS
LSTM DEEP LEARNING MODEL: 
Training Specifications: Look back = 60, activation = relu, optimizer = Adam, Loss = mean squared error.
Evaluation metric: RMSE = 0.002
 
RANDOM FOREST MODEL:
Training Specifications: Random Forest regressor, random_state = 1
Evaluation metric: RMSE = 0.02
 

XGBOOST MODEL:
Training Specifications: Learning rate = 0.1, alpha = 10, N_estimators = 200
Evaluation metric: RMSE =0.08
 
Model chosen:
While evaluating the results of the models and comparing them, it was found as we can see that LSTM is the winner model, with lowest error RMSE =  0.002 and high correlation of predicted vs real values. But this was not sufficient to consider it the winner. Further investigation was done and it was found that LSTM results are not consistent in real time and maybe it needs more refactoring for the data. However, the tree based algorithms based on machine learning showed consistency in performance and consistency in the results and evaluation, and thus Random Forest algorithm was chosen in this case although it has less accuracy but it has shown very good results.
 

5  DAYS RANGE PREDICTION:
It is common in financial analysis on crypto market to have a 5 days prediction for the price of the coin trying to predict. And in this case this can be done through adding the predicted price as a feature in the 60 days and in this way, the 5 days prediction is possible. This has been added to the project and can be done for the 3 algorithms chosen. The output is a csv file containing the price predicted for the 3 algorithms. 
 

