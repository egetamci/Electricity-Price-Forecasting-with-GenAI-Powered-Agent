# Electricity-Price-Forecasting-with-GenAI-Powered-Agent

# LSTM Electricity Price day-ahead Forecasting 

This technical documentation provides a detailed guide to setting up, running, and understanding the **Electricity Price Forecasting System**. The system collects, processes, and forecasts electricity price data using a **LSTM** model.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
  - [Key Components](#key-components)
  - [Functions and Logic](#functions-and-logic)
- [File Descriptions](#file-descriptions)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Introduction

The **Electricity Price Forecasting System** is a machine learning-based forecasting tool designed to predict electricity prices. It collects data directly from the Hungarian Power Exchange (HUPX) website, processes the data, and uses a **LSTM** (Long Short-Term Memory) model to generate day-ahead forecasts.

---

## Features

- **Data Collection**: Automatically fetches electricity price data from HUPX market at least using a 3-month time window (90 days).
- **Data Preprocessing**: Cleans, formats, and prepares data for the forecasting model. High correlation features have been removed.
- **Forecasting**:
  - Utilizes **LSTM** for time series predictions.
  - Forecasts electricity prices for the next 24×4 time steps of a day. Data is provided every 15 minutes.
- **Visualization**: Interactive visualizations of actual vs predicted prices using Plotly.
- **Validation**: Ensures the generated forecasts meet predefined accuracy and format criteria.

---

## Requirements

- **Python** 3.8 or later
- pandas
- requests
- beautifulsoup4
- langchain
- openai
- tensorflow
- scikit-learn
- numpy
- plotly
- langchain
- lxml
- matplotlib
- python-dotenv

---

## Installation


1. **Install Dependencies**:
   Install all required Python libraries listed above.
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up OpenAI API Key**:
   - Create a `.env` file in the project root directory.
   - Add the following line with your OpenAI API key:
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     ```

---

## Usage

### Running the System

1. **Start the System**:
   Run the Python script to initialize the forecasting system.
   ```bash
   python test.py
   ```

2. **Interact with the System**:
   Provide commands in natural language, such as:
   - "Fetch data from 2024-01-01 to 2024-01-10."
   - "Generate predictions."
   - "Validate forecasts."
   - Use `exit` or `quit` to close the system.

---

## Code Overview

### Key Components

1. **Data Collection**:
   - `fetch_hupx_data(date)`: Fetches hourly electricity price data from HUPX for a given date.
   - `collect_hupx_data(start_date, end_date)`: Collects and consolidates data over a date range.

2. **Data Preprocessing**:
   - Cleans missing or invalid values.
   - Scales features using `MinMaxScaler`.

3. **Forecasting**:
   - `create_model()`: Builds and trains a **LSTM** model.
   - `forecasting()`: Generates predictions for the next 24*4 # 1 day forecast (Data Provided Every 15 Minutes).

4. **Validation**:
   - Ensures all forecasts are numeric, within realistic value ranges (e.g., 0-500).

5. **Visualization**:
   - Compares actual and predicted prices using interactive Plotly charts.

---

## File Descriptions

- `test.py`: Main script to run the system.
- `requirements.txt`: List of required Python libraries.
- `README.md`: Technical documentation for the project.

---

## Examples

### Collecting Data

Command:
```bash
Fetch data from 2024-01-01 to 2024-01-10.
```
Output:
- Data is collected and saved as `hupx_data_2024-01-01_to_2024-01-10.csv`.

### Generating Predictions

Command:
```bash
Generate predictions.
```
Output:
- Predictions are generated and saved as `rnn_predictions.csv`.
- Visualization is displayed comparing actual vs predicted prices.

---

## Troubleshooting

1. **API Key Issues**:
   - Ensure your OpenAI API key is correctly set in the `.env` file.

2. **Invalid Date Range**:
   - Dates must be in the `YYYY-MM-DD` format.
   - Ensure the requested dates are within a valid range.

3. **Dependencies Not Installed**:
   - Ensure all libraries listed in `requirements.txt` are installed:
     ```bash
     pip install -r requirements.txt
     ```

4. **No Data Found**:
   - Check the HUPX website for data availability on the specified dates.

---

## Future Enhancements

- Incorporating additional machine learning models for comparison.
- Adding support for hybrid model combinations (e.g., LSTM + ARIMA).
- Expanding data sources for more robust forecasting.

---
