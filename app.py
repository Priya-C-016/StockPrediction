import gdown
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Downloading the dataset from Google Drive using gdown
file_id = '1Cx2e2SV6uJYMesplAawrORfk38grcdx4'
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, 'all_stocks_5yr.csv', quiet=False)

# Load data into pandas dataframe
data = pd.read_csv('all_stocks_5yr.csv', delimiter=',', on_bad_lines='skip')
data['date'] = pd.to_datetime(data['date'])

# Sidebar Filters for companies and plots
st.sidebar.title("Stock Analysis Filters")
companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'EBAY', 'CSCO', 'IBM']
selected_companies = st.sidebar.multiselect("Select Companies to Analyze", companies, default=companies)
view_plots = st.sidebar.multiselect("Select Plots to Display", ['Open vs Close', 'Volume'], default=['Open vs Close'])

# Title and dataset preview
st.title("Stock Market Analysis")
st.write("Dataset Overview")
st.write(data.head())

# Filter data for selected companies
filtered_data = data[data['Name'].isin(selected_companies)]

# Update graphs and tables based on selected companies
def plot_open_vs_close():
    st.subheader("Open vs Close Prices Over Time")
    plt.figure(figsize=(15, 8))
    for index, company in enumerate(selected_companies, 1):
        plt.subplot(3, 3, index)
        company_data = filtered_data[filtered_data['Name'] == company]
        plt.plot(company_data['date'], company_data['close'], label="Close", color="r", marker="+")
        plt.plot(company_data['date'], company_data['open'], label="Open", color="g", marker="^")
        plt.title(company)
        plt.legend()
        plt.tight_layout()
    st.pyplot(plt)

def plot_volume():
    st.subheader("Volume Over Time")
    plt.figure(figsize=(15, 8))
    for index, company in enumerate(selected_companies, 1):
        plt.subplot(3, 3, index)
        company_data = filtered_data[filtered_data['Name'] == company]
        plt.plot(company_data['date'], company_data['volume'], color="purple", marker="*")
        plt.title(f"{company} Volume")
        plt.tight_layout()
    st.pyplot(plt)

# Stock Price Prediction for a selected company
def stock_price_prediction(company):
    st.subheader(f"Stock Price Prediction for {company}")
    company_data = data[data['Name'] == company]
    company_data['date'] = pd.to_datetime(company_data['date'])

    # Feature Engineering: Using previous 'n' days' closing prices to predict next day's close
    n_days = 60  # Number of days used to predict the next day's price

    # Prepare the data for training the model
    close_data = company_data['close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data.reshape(-1, 1))

    X = []
    y = []

    for i in range(n_days, len(scaled_data)):
        X.append(scaled_data[i-n_days:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

    # Evaluation Metrics
    mse = mean_squared_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred)
    mae = mean_absolute_error(scaler.inverse_transform(y_test.reshape(-1, 1)), y_pred)
    rmse = np.sqrt(mse)

    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Show the predictions along with the real stock prices
    prediction_df = pd.DataFrame({
        'Date': company_data['date'].iloc[len(company_data) - len(y_test):].values,
        'Real Close Price': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
        'Predicted Close Price': y_pred.flatten()
    })

    st.write("Prediction vs Real Data")
    st.write(prediction_df)

    # Plot the real vs predicted stock prices
    plt.figure(figsize=(10, 6))
    plt.plot(prediction_df['Date'], prediction_df['Real Close Price'], label='Real Close Price', color='blue')
    plt.plot(prediction_df['Date'], prediction_df['Predicted Close Price'], label='Predicted Close Price', color='red', linestyle='--')
    plt.title(f'{company} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# Show plots based on user selection
if 'Open vs Close' in view_plots:
    plot_open_vs_close()

if 'Volume' in view_plots:
    plot_volume()

# Show stock price prediction for the first selected company
if selected_companies:
    stock_price_prediction(selected_companies[0])
