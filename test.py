
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense,LSTM
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.graph_objects as go
import re
from dotenv import load_dotenv

import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------
load_dotenv()  # Loads variables from a .env file into environment

# -------------------------------------------------------------
# Suppress Warning Messages
# -------------------------------------------------------------
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# Set OpenAI API Key
# -------------------------------------------------------------
# **Important:** Ensure that you have a .env file with the line:
# OPENAI_API_KEY=your_actual_api_key_here

openai_api_key = os.environ["OPENAI_API_KEY"] = "x"

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set it in the .env file.")
os.environ["OPENAI_API_KEY"] = openai_api_key

# -------------------------------------------------------------
# Global Variables
# -------------------------------------------------------------
collected_data = None   # Stores the collected data
predictions = None      # Stores the prediction results
price_series = None     # Stores the actual price series

# Define forecast horizon
FORECAST_INTERVALS_NUM = 24*4 # 1 day forecast (Data Provided Every 15 Minutes)

# Define input feature columns (excluding the target and specified columns)
# The vectors were selected based on their correlation with each other.
FEATURE_COLUMNS = [
    'Buy traded volume (MW)',
    #'Sell traded volume (MW)',
    'VWAP of the last trading hour (EUR/MWh)',
    'Volume weighted average price (EUR/MWh)'
]

TARGET_COLUMN = 'Volume weighted average price (EUR/MWh)'

# -------------------------------------------------------------
# Function to Parse Dates from User Input using OpenAI
# -------------------------------------------------------------
def parse_dates_with_openai(user_input):
    """
    Extracts 'start_date' and 'end_date' in 'YYYY-MM-DD' format from user input using OpenAI.
    Returns a tuple (start_date, end_date). If parsing fails, returns (None, None).
    """
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = (
        f"The user provided the following text: '{user_input}'. "
        "From this text, extract the 'start_date' and 'end_date' in 'YYYY-MM-DD' format. "
        "Example output: {'start_date': '2020-01-01', 'end_date': '2020-12-31'}"
    )
    response = llm.predict(prompt)
    try:
        # Safely parse the returned string as a dictionary
        dates = eval(response)
        if isinstance(dates, dict):
            return dates.get("start_date"), dates.get("end_date")
        else:
            print("Unexpected response format from OpenAI.")
            return None, None
    except Exception as e:
        print(f"Date parsing error: {e}")
        return None, None

# -------------------------------------------------------------
# Function to Clean Time Interval Strings
# -------------------------------------------------------------
def clean_time_interval(interval):
    """
    Cleans the time interval string by removing any trailing uppercase letters (e.g., 'A', 'B').
    If the format is incorrect, returns NaN.
    """
    match = re.match(r'^(\d{2}:\d{2}-\d{2}:\d{2})', interval)
    if match:
        return match.group(1)
    else:
        return np.nan  # Or handle as appropriate

# -------------------------------------------------------------
# Function to Fetch Data from HUPX Website for a Given Date
# -------------------------------------------------------------
def fetch_hupx_data(date):
    """
    Fetches electricity price data from the HUPX website for a specified date.
    Returns a pandas DataFrame if successful, otherwise None.
    """
    base_url = f"https://hupx.hu/en/market-data/id/market-data?date={date}"
    try:
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")
        tables = soup.find_all("table")
        if tables:
            # Parse the first table found on the page
            df = pd.read_html(str(tables[0]))[0]
            print(f"Data fetched for date: {date}")
            
            # -------------------------------
            # Clean Time Interval Column
            # -------------------------------
            # Get the name of the first column (likely 'Unnamed: 0' or similar)
            time_interval_column = df.columns[0]
            
            # Remove trailing uppercase letters (A, B, etc.) from time interval values
            df[time_interval_column] = df[time_interval_column].str.replace(r'[A-Z]$', '', regex=True)
            
            # Additional Cleanup: Ensure time intervals are correctly formatted
            df[time_interval_column] = df[time_interval_column].apply(clean_time_interval)
            
            # Drop rows with NaN in the time interval column
            df = df.dropna(subset=[time_interval_column])
            
            return df
        else:
            print(f"No table found for date: {date}")
            return None
    except Exception as e:
        print(f"Error fetching data for {date}: {e}")
        return None

# -------------------------------------------------------------
# Function to Collect Data Over a Date Range
# -------------------------------------------------------------
def collect_hupx_data(start_date, end_date):
    """
    Collects electricity price data from HUPX over a specified date range.
    Returns a concatenated pandas DataFrame of all fetched data. If no data is fetched, returns None.
    
    """
    global end_date_dt
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    all_data = []

    while current_date <= end_date_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        df = fetch_hupx_data(date_str)
        if df is not None:
            df['Date'] = date_str  # Add a Date column for reference
            all_data.append(df)
        current_date += timedelta(days=1)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return None

# -------------------------------------------------------------
# Data Collection Tool for LangChain Agent
# -------------------------------------------------------------
def data_collection_tool(inputs):
    """
    Processes user input to determine date range, collects corresponding HUPX data,
    and saves the data to a CSV file.
    Returns a status message.
    """
    global collected_data, price_series

    # Parse dates using OpenAI
    start_date, end_date = parse_dates_with_openai(inputs)
    if not start_date or not end_date:
        return "Could not parse dates. Please provide a valid date range."

    # Define valid date range for data collection
    valid_start_date = datetime.strptime("2019-12-14", "%Y-%m-%d")
    valid_end_date = datetime.strptime("2025-01-26", "%Y-%m-%d")

    try:
        start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Invalid date format. Dates must be in 'YYYY-MM-DD' format."

    # Validate the date range
    if start_date_obj < valid_start_date or start_date_obj > valid_end_date:
        return "Please enter a valid date range."
    if end_date_obj < valid_start_date or end_date_obj > valid_end_date:
        return "Please enter a valid date range."
    if start_date_obj > end_date_obj:
        return "Please enter a valid date range."
   # Check if the selected date range is long enough for the model
    minimum_days_required = 90 + 1  # 90 days for the window size and at least 1 day for the forecast interval
    delta = end_date_obj - start_date_obj
    
    if delta.days < minimum_days_required:
        return f"Date range is too short. Please to do forecasting select a date range of at least 91 days, starting before {end_date}."

    # Collect the data
    collected_data = collect_hupx_data(start_date, end_date)

    if collected_data is None:
        return f"No data could be fetched for the range {start_date} to {end_date}."

    # -------------------------------
    # Additional Cleanup
    # -------------------------------
    time_interval_column = collected_data.columns[0]
    collected_data[time_interval_column] = collected_data[time_interval_column].apply(clean_time_interval)
    collected_data = collected_data.dropna(subset=[time_interval_column])
    collected_data['Time'] = collected_data['Unnamed: 0'].str.split('-').str[1]

    # Combine the Time and Date columns
    collected_data['Combined'] = collected_data['Time'] + " " + collected_data['Date']
    collected_data = collected_data.drop(columns=['Unnamed: 0','Date', 'Time'])
    collected_data['Combined'] = pd.to_datetime(collected_data['Combined'], format='%H:%M %Y-%m-%d')
    collected_data= collected_data.set_index(['Combined'], drop=True)
    collected_data = collected_data[FEATURE_COLUMNS]





    # Convert relevant columns to numeric, handle missing or invalid values
    for col in FEATURE_COLUMNS:
        if col in collected_data.columns:
            collected_data[col] = (
                collected_data[col]
                .replace('-', np.nan)
                .astype(float)
            )
        else:
            print(f"Warning: Expected column '{col}' not found in the data.")

    # Handle missing values by imputing with forward fill, then backward fill, then mean
    for col in FEATURE_COLUMNS:
        if col in collected_data.columns:
            collected_data[col].fillna(method='ffill', inplace=True)
            collected_data[col].fillna(method='bfill', inplace=True)
            collected_data[col].fillna(collected_data[col].mean(), inplace=True)

    # After imputation, check for any remaining NaNs
    if collected_data.isnull().any().any():
        return "Data contains NaNs even after imputation. Please check the data source."

    # Extract target series
    price_series = collected_data[TARGET_COLUMN]
    #collected_data = collected_data[FEATURE_COLUMNS]
    # Save data to CSV
    filename = f"hupx_data_{start_date}_to_{end_date}.csv"
    collected_data.to_csv(filename, index=False)
    #print(pd.read_csv(filename))
 # Print the content of the CSV file to the console


    return f"Data successfully collected for {start_date} to {end_date} and saved to '{filename}'"

###############################################################
def create_sequences(data, n_input, n_out):
    # Initialize a Min-Max Scaler to scale the data between 0 and 1
    scaler = MinMaxScaler()
    
    # Get the index of the target column (the column to predict)
    target_idx = data.columns.get_loc(TARGET_COLUMN)
    
    # Scale the entire dataset
    scaled_data = scaler.fit_transform(data)
    
    # Initialize lists to hold input sequences (X) and target values (y)
    X, y = [], []
    
    # Loop through the dataset to create domain "X" and range "y" sequences of size `window_size`
    X, y = [], []
    for i in range(len(combined_scaled)):
        end_ix = i + n_input
        out_end_ix = end_ix + n_out
        if out_end_ix > len(combined_scaled):
            break
        seq_x = combined_scaled[i:end_ix, :-1]  # All features except target
        seq_y = combined_scaled[end_ix:out_end_ix, -1]  # Only target
    
    # Calculate the split point to separate training and testing data
    split = len(X) - FORECAST_INTERVALS_NUM
    
    # Split the sequences into training and testing sets
    X_train, X_test = seq_x[:split], seq_x[split:]
    y_train, y_test = seq_y[:split], seq_y[split:]
    
    # Return the arrays for training and testing, along with the scaler and target index
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test), scaler, target_idx



def create_model(X_train, y_train, X_test, y_test):
    # Define a sequential model with an LSTM layer
    model = Sequential([
        # Add an LSTM layer with 64 units and 'tanh' activation function
        LSTM(64,input_shape=(X_train.shape[1], X_train.shape[2])),
        # Add a Dense layer with 32 neurons and 'tanh' activation
        Dense(32, activation='tanh'),
        # Add an output layer with 1 neuron for the target variable
        Dense(FORECAST_INTERVALS_NUM)  
    ])

    # Compile the model with Adam optimizer, Huber Loss, and mean absolute error as a metric
    #model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.compile(optimizer='adam', loss=Huber(), metrics=['mae'])


    # Define a checkpoint to save the best model based on validation loss
    checkpoint = ModelCheckpoint('model.keras', save_best_only=True, monitor='val_loss', mode='min')

    # Define early stopping to monitor the validation loss and stop training when it doesn't improve
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Return the trained model
    return model


def forecasting(data, n_input, n_out):
    # Returns scaled training/testing data and additional info like the scaler and target index.
    X_train, X_test, y_train, y_test, scaler, target_idx = create_sequences(data, n_input, n_out=)
    
    # Create and train the model using training data, and validate it using testing data.
    model = create_model(X_train, y_train, X_test, y_test)
    
    # Use the trained model to predict values for the test set.
    predictions = model.predict(X_test)
    
    # Reverse the scaling (inverse transformation) for the test targets to bring them back to original scale.
    y_test_original = scaler.inverse_transform(
        [[0] * target_idx + [val] + [0] * (data.shape[1] - target_idx - 1) for val in y_test]
    )[:, target_idx]
    
    # Reverse the scaling for the predictions as well, to match the original scale.
    predictions = scaler.inverse_transform(
        [[0] * target_idx + [val] + [0] * (data.shape[1] - target_idx - 1) for val in predictions.flatten()]
    )[:, target_idx]
    
    # Return the original test values and predictions in the original scale.
    return predictions, y_test_original



def prediction_tool(inputs):
    """
    Uses the collected data to generate forecasts using the LSTM model.
    Saves the forecasts to a CSV file and visualizes them using Plotly.
    Returns a status message.
    """
    global collected_data
    if collected_data is None:
        return "No predictions can be made. Please collect the data first."
    predictions,y_test_original = forecasting(collected_data, n_input = 24*4*90, n_out = 24*4)
    # Prepare sequences for training and testing using a 3-month time window (90 days).

    forecast = predictions
    actual_prices = y_test_original
    actual_steps = list(range(len(actual_prices)))
    forecast_steps = list(range(len(forecast)))

    # Calculate the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)
    mae = np.mean(np.abs(actual_prices - forecast))
    rmse = np.sqrt(np.mean((actual_prices - forecast) ** 2))

    # Create an interactive plot
    fig = go.Figure()

# Plot Actual Prices
    fig.add_trace(go.Scatter(
      x=actual_steps,
      y=actual_prices,
      mode='lines+markers',  # Show markers along with lines
      name='Actual Prices',
      line=dict(dash='solid', color='orange', width=4),  # Solid line with thicker width
      marker=dict(size=8, color='orange', symbol='circle')  # Add markers for emphasis
    ))

# Plot LSTM Forecast
    fig.add_trace(go.Scatter(
      x=forecast_steps,
      y=forecast,
      mode='lines+markers',
      name='LSTM Forecast',
      line=dict(color='royalblue', width=4),
      marker=dict(size=8, color='royalblue', symbol='triangle-up')
    ))

# Add annotations for MAE and RMSE
    annotation_text = (
      f"<b>LSTM Model Performance</b><br>"
      f"<i>Mean Absolute Error (MAE):</i> <b>{mae:.2f}</b><br>"
      f"<i>Root Mean Squared Error (RMSE):</i> <b>{rmse:.2f}</b>"
    )

    fig.add_annotation(
    x=0.95, y=0.95, xref='paper', yref='paper',
    text=annotation_text,
    showarrow=False,
    font=dict(size=14, color="black", family="Arial Bold"),
    bgcolor="lightgrey",
    opacity=0.7,
    align="right"
)

# Update layout with more professional titles, axis labels, and color scheme 
    if isinstance(end_date_dt, datetime):
    # If end_date_dt is a datetime object, there is no need to use strptime.
      date_obj = end_date_dt
    elif isinstance(end_date_dt, str):
    #If end_date_dt is a string, let's convert it to a datetime object using strptime.
       date_obj = datetime.strptime(end_date_dt, "%Y-%m-%d %H:%M:%S")
    else:
       raise ValueError("Invalid type for end_date_dt")

# Use timedelta to add one day
    next_day = date_obj + timedelta(days=1)

# Get the new date in just the date format
    next_day_str = next_day.strftime("%Y-%m-%d")
    fig.update_layout(
     title=f"<b style='font-size:24px; color: #1E88E5;'>LSTM-Based Electricity Price Forecast</b><br><i style='font-size:18px; color: #555555;'>Predicted Prices for {next_day_str}</i>",
     title_x=0.5,  # Center the title
     xaxis_title="<b>Time Step (Hours)</b>",
     yaxis_title="<b>Price (EUR/MWh)</b>",
     #legend_title="<b>Legend</b>",
     template="plotly_dark",  # Dark mode for modern look
     plot_bgcolor='rgb(32, 32, 32)',  # Set dark background
     paper_bgcolor='rgb(32, 32, 32)',
     xaxis=dict(
       showgrid=True, 
       gridwidth=1, 
       gridcolor='rgb(85, 85, 85)', 
       zeroline=False,
       tickmode='array',  # Specify custom tick positions
       tickvals=list(range(0, 95, 4)),  # Position ticks at every 4th data point
       ticktext=[str(i) for i in range(0, 24)],  # Label ticks from 0 to 23
       title="<b>Time Step (Hours)</b>",
       range=[0, 95]  # Full range of data points
     ),
     yaxis=dict(
        showgrid=True, 
        gridwidth=1, 
        gridcolor='rgb(85, 85, 85)', 
        zeroline=False
     )
    )

# Display the interactive plot
    fig.show()

    return f"Forecast successfully generated and saved to 'lstm_predictions.csv'."
    


########################################################
def validate_predictions(predictions):
    """
    Validates the forecasted predictions.
    Checks:
      - Predictions must be a list-like structure.
      - Predictions list must not be empty.
      - All values must be numeric.
      - All values must be between 0 and 500.
    Returns a validation message.
    """
    if not isinstance(predictions, (list, pd.Series, np.ndarray)):
        return "Predictions must be in a list-like structure."
    if len(predictions) == 0:
        return "Prediction list cannot be empty."
    for idx, value in enumerate(predictions):
        if not isinstance(value, (float, int, np.float32, np.float64)):
            return f"All predictions must be numeric. Invalid value at index {idx}: {value}"
        if not (0 <= value <= 500):
            return f"Prediction value out of range at index {idx}: {value}. Values must be between 0 and 500."
    return f"Predictions are valid and verified. Total predictions: {len(predictions)}."

# -------------------------------------------------------------
# Validation Tool for LangChain Agent
# -------------------------------------------------------------
def validation_tool(inputs):
    """
    Validates the generated forecasts using the validate_predictions function.
    Returns a validation message.
    """
    global predictions
    if predictions is None:
        return "No forecasts to validate. Please generate predictions first."
    return validate_predictions(predictions)


########################################################



# -------------------------------------------------------------
tools = [
    Tool(
        name="Data Collection",
        func=data_collection_tool,
        description="Collect electricity price data for a specified date range."
    ),
        Tool(
        name="Prediction",
        func=prediction_tool,
        description="Generate 1-day (24-hour) electricity price forecasts using an LSTM model with multiple input features."
    ),
    Tool(
        name="Validation",
        func=validation_tool,
        description="Validate the generated 1-day (24-hour) forecasts."
    )
    
]

# Initialize the OpenAI language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Initialize the LangChain agent with the defined tools
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# -------------------------------------------------------------
# User Interaction Loop
# -------------------------------------------------------------
print("System initialized. You can provide natural language commands to operate.")

while True:
    user_input = input("\nCommand: ")
    if any(exit_word in user_input.lower() for exit_word in ["exit", "quit", "çıkış"]):
        print("Exiting the system...")
        break
    else:
        try:
            result = agent.run(user_input)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
