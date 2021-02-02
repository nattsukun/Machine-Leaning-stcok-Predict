# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression

# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

# yahoo finance is used to fetch data
import yfinance as yf

# Read data
Df = yf.download('SCB.BK', '2015-01-01', '2021-02-15', auto_adjust=True)
#GC=F    = gold
#BTC-USD   = btc
#USDT  = USDT-USD
# Only keep close columns
Df = Df[['Close']]

# Drop rows with missing values
Df = Df.dropna()

# Plot the closing price of GLD
Df.Close.plot(figsize=(20, 14),color='r')
plt.ylabel("SCB-STOCK ETF Prices")
plt.title("SCB-STOCK ETF Price Series")
plt.show()

# Define explanatory variables
Df['S_3'] = Df['Close'].rolling(window=3).mean()
Df['S_9'] = Df['Close'].rolling(window=9).mean()
Df['next_day_price'] = Df['Close'].shift(-1)

Df = Df.dropna()
X = Df[['S_3', 'S_9']]

# Define dependent variable
y = Df['next_day_price']
print('nexdayprice')
print(y)
# Split the data into train and test dataset
t = .8
t = int(t*len(Df))

# Train dataset
X_train = X[:t]
y_train = y[:t]

# Test dataset
X_test = X[t:]
y_test = y[t:]
print('train')
# Create a linear regression model
linear = LinearRegression().fit(X_train, y_train)
print("Linear Regression model")
print("SCB-STOCK ETF Price (y) = %.2f * 3 Days Moving Average (x1) \
+ %.2f * 9 Days Moving Average (x2) \
+ %.2f (constant)" % (linear.coef_[0], linear.coef_[1], linear.intercept_))
print('endtrain')
# Predicting the SCB-STOCK ETF prices
predicted_price = linear.predict(X_test)
predicted_price = pd.DataFrame(
    predicted_price, index=y_test.index, columns=['price'])
predicted_price.plot(figsize=(20, 14))
y_test.plot()
plt.legend(['predicted_price', 'actual_price'])
plt.ylabel("SCB-STOCK ETF Price")
plt.show()


print('show grapth2')
# R square
r2_score = linear.score(X[t:], y[t:])
float("{0:.2f}".format(r2_score))
print('r2score')
print(r2_score)
data = pd.DataFrame()

data['price'] = Df[t:]['Close']
data['predicted_price_next_day'] = predicted_price
data['actual_price_next_day'] = y_test
data['data_returns'] = data['price'].pct_change().shift(-1)

data['signal'] = np.where(data.predicted_price_next_day.shift(1) < data.predicted_price_next_day,1,0)

data['strategy_returns'] = data.signal * data['data_returns']
((data['strategy_returns']+1).cumprod()).plot(figsize=(20,14),color='g')
plt.ylabel('Cumulative Returns')
plt.show()

print('show signal')
'Sharpe Ratio %.2f' % (data['strategy_returns'].mean()/data['strategy_returns'].std()*(252**0.5))

data = yf.download('SCB.BK', '2015-01-01', '2021-02-15', auto_adjust=True)
data['S_3'] = data['Close'].rolling(window=3).mean()
data['S_9'] = data['Close'].rolling(window=9).mean()
data = data.dropna()
XX = data[['S_3', 'S_9']]
data['predicted_data_price'] = linear.predict(XX)
data['signal'] = np.where(data.predicted_data_price.shift(1) < data.predicted_data_price,"Buy","No Position")
data.to_csv('dataframeSCB-BK.csv')
#data.tail(7)

