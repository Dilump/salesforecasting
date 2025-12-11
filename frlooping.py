import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from xgboost import XGBRegressor
import joblib

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('train.csv')

# --- 1. Data Preparation and Feature Engineering ---
# Sort by date for proper time-series split
df['date'] = pd.to_datetime(df['date'])
df.sort_values(by='date', inplace=True)

# Extract time-based features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)

# Monthly seasonality features (using sin/cos)
df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))

# Optional: Lag features (more robust for time-series)
# Calculate a 7-day rolling average of sales for each item
df['lag_7'] = df.groupby('item')['sales'].transform(lambda x: x.shift(7).rolling(7).mean())
df.dropna(inplace=True) # Drop rows with NaN from rolling average

# Drop original date and unnecessary features
df.drop('date', axis=1, inplace=True)

# Remove extreme outliers (based on domain knowledge if possible)
df = df[df['sales'] < 140]

# Get a list of unique items
items = df['item'].unique()

# Dictionary to store trained models
item_models = {}

# --- 2. Model Training and Evaluation ---
for item_id in items:
    print(f"Training model for Item {item_id}...")

    # Filter data for this specific item
    item_data = df[df['item'] == item_id]

    # Time-series data split (80% for training, 20% for validation)
    split_point = int(len(item_data) * 0.8)
    train_data = item_data.iloc[:split_point]
    val_data = item_data.iloc[split_point:]

    # Define features and target
    features = ['item', 'month', 'day', 'weekday', 'weekend', 'm1', 'm2', 'lag_7']
    target = 'sales'

    X_train, Y_train = train_data[features], train_data[target]
    X_val, Y_val = val_data[features], val_data[target]

    # Train XGBoost model (no scaling needed for tree-based models)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, Y_train)

    # Predictions
    val_preds = model.predict(X_val)
    val_preds[val_preds < 0] = 0 # Ensure predictions are not negative

    # Evaluation
    print(f"Validation MAE for Item {item_id}: {mae(Y_val, val_preds):.2f}\n")

    # Save the trained model to a file
    model_filename = f'model_item_{item_id}.joblib'
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}\n")

    # Visual representation
    plt.figure(figsize=(10,5))
    plt.plot(Y_val.values, label='Actual Sales', marker='o')
    plt.plot(val_preds, label='Predicted Sales', marker='x')
    plt.title(f'Item {item_id} - Actual vs Predicted Sales')
    plt.xlabel('Validation Samples')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()
    
# --- 3. Prediction Function for Production Use ---
def predict_sales_for_item(item_id, date_to_predict):
    """
    Predicts sales for a given item on a future date.
    
    Args:
        item_id (int): The ID of the item.
        date_to_predict (datetime): The date for the prediction.
        
    Returns:
        float: The predicted sales quantity.
    """
    # Load the trained model for the specific item
    try:
        model = joblib.load(f'model_item_{item_id}.joblib')
    except FileNotFoundError:
        print(f"Error: No model found for Item {item_id}.")
        return None

    # Create a DataFrame with the features for the prediction date
    future_data = pd.DataFrame([{
        'item': item_id,
        'month': date_to_predict.month,
        'day': date_to_predict.day,
        'weekday': date_to_predict.weekday(),
        'weekend': 1 if date_to_predict.weekday() >= 5 else 0,
        'm1': np.sin(date_to_predict.month * (2 * np.pi / 12)),
        'm2': np.cos(date_to_predict.month * (2 * np.pi / 12)),
        # Note: Lag features require historical data up to the prediction date.
        # This is a simplification. In a real system, you'd get this from your DB.
        'lag_7': 10 # Example value, you'd need to compute this from real data
    }])
    
    # Make the prediction
    prediction = model.predict(future_data)
    
    # Return the prediction, ensuring it's not negative
    return max(0, prediction[0])

# Example usage of the prediction function
from datetime import datetime
future_date = datetime(2025, 1, 15)
item_to_predict = 10 # Example item ID

predicted_sales = predict_sales_for_item(item_to_predict, future_date)
print(f"\nPredicted sales for Item {item_to_predict} on {future_date.date()}: {predicted_sales:.2f}")