import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# --- Prediction Function (reuse from your script) ---
def predict_sales_for_item(item_id, date_to_predict):
    try:
        model = joblib.load(f'model_item_{item_id}.joblib')
    except FileNotFoundError:
        return None
    
    features = pd.DataFrame([{
        'item': item_id,
        'month': date_to_predict.month,
        'day': date_to_predict.day,
        'weekday': date_to_predict.weekday(),
        'weekend': 1 if date_to_predict.weekday() >= 5 else 0,
        'm1': np.sin(date_to_predict.month * (2 * np.pi / 12)),
        'm2': np.cos(date_to_predict.month * (2 * np.pi / 12)),
        'lag_7': 10  # placeholder; ideally calculated from real data
    }])

    pred = model.predict(features)
    return max(0, pred[0])

# --- Streamlit UI ---
st.title("📊 Sales Prediction Dashboard")
st.write("Predict future sales for different items using your trained XGBoost models.")

# Sidebar controls
st.sidebar.header("🔧 Prediction Controls")

# Example item IDs (you can change this list based on your data)
item_ids = list(range(1, 51))
item_id = st.sidebar.selectbox("Select Item ID:", item_ids)

# Date picker
selected_date = st.sidebar.date_input("Select Date:", datetime(2025, 1, 15))

# Predict button
if st.sidebar.button("Predict Sales"):
    result = predict_sales_for_item(item_id, selected_date)
    if result is not None:
        st.success(f"✅ Predicted Sales for Item {item_id} on {selected_date}: **{result:.2f} units**")
    else:
        st.error("No trained model found for this item. Please train it first.")

# Optional: display chart placeholder
st.subheader("📈 Future Prediction Trend (Demo)")
future_dates = pd.date_range(selected_date, periods=7)
predictions = [predict_sales_for_item(item_id, d) for d in future_dates]
chart_data = pd.DataFrame({'Date': future_dates, 'Predicted Sales': predictions})
st.line_chart(chart_data.set_index('Date'))
