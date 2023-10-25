import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load data
symbol = "^NSEI"  # NIFTY50
df = yf.download(symbol, start="2010-01-01", end="2020-01-01")

# Feature Engineering: Create the features you want to use. Here, we use simple ones.
df['Open-Close'] = (df.Open - df.Close)/df.Open
df['High-Low'] = (df.High - df.Low)/df.Low
df['percent_change'] = df['Adj Close'].pct_change()
df['std_5'] = df['percent_change'].rolling(5).std()
df['ret_5'] = df['percent_change'].rolling(5).mean()
df.dropna(inplace=True)

# Target Variable: Predicting whether the stock will go up or down. Here, 1 means "up" and 0 means "down".
df['target'] = df['percent_change'].apply(lambda x: 1 if x > 0 else 0)

# Define your feature set and target variable
feature_set = df[['Open-Close', 'High-Low', 'std_5', 'ret_5']]
target_set = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_set, target_set, test_size=0.2, random_state=42)

# Initialize and train the classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Check the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

# Backtesting
initial_portfolio = 100000  # Starting with 100k
current_portfolio = initial_portfolio
in_position = False

# Tracking the buy and hold strategy
buy_and_hold_start_value = None
buy_and_hold_end_value = None

# Simulate the trading - we will use a different approach here to avoid indexing issues
test_indices = X_test.index.tolist()  # Convert indices to a list for proper iteration

for i in range(len(test_indices)):
    current_index = test_indices[i]
    current_price = df.loc[current_index]['Close']
    
    if buy_and_hold_start_value is None:
        buy_and_hold_start_value = current_price  # start value for buy and hold

    buy_and_hold_end_value = current_price  # updating end value for buy and hold

    if y_pred[i] == 1 and not in_position:  # We are accessing y_pred directly with i
        # We buy at the close price
        in_position = True
        buy_price = current_price
        shares_bought = current_portfolio / buy_price

    elif y_pred[i] == 0 and in_position:  # We are accessing y_pred directly with i
        # We sell at the close price
        in_position = False
        sell_price = current_price
        sale_proceeds = shares_bought * sell_price
        current_portfolio = sale_proceeds

# If we're still holding the stock, sell it at the last known price
if in_position:
    current_portfolio = shares_bought * buy_and_hold_end_value

# Calculate the strategy's returns as a percentage
strategy_return = (current_portfolio - initial_portfolio) / initial_portfolio * 100

# Calculate the buy and hold strategy's returns as a percentage
buy_and_hold_return = (buy_and_hold_end_value - buy_and_hold_start_value) / buy_and_hold_start_value * 100

# Print the results
print(f"Strategy Return: {strategy_return}%")
print(f"Buy and Hold Return: {buy_and_hold_return}%")