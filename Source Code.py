import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===============================================================
# CONFIG
# ===============================================================
DATA_PATH = r"C:\Users\shubh\Downloads\ltsm"
replacebg_file = os.path.join(DATA_PATH, "replacebg_.csv")
openaps_file   = os.path.join(DATA_PATH, "openaps_.csv")

# ===============================================================
# LOAD DATA
# ===============================================================
replacebg = pd.read_csv(replacebg_file)
openaps   = pd.read_csv(openaps_file)

# ===============================================================
# DETECT GLUCOSE COLUMN
# ===============================================================
def find_glucose_column(df):
    possible_cols = ["glucose", "bg", "sgv", "cgm", "sensor"]
    for col in df.columns:
        if any(key in col.lower() for key in possible_cols):
            return col
    raise ValueError("No glucose-related column found.")

for dfX in [replacebg, openaps]:
    gcol = find_glucose_column(dfX)
    dfX.rename(columns={gcol: "glucose"}, inplace=True)

# Merge and clean
df = pd.concat([replacebg, openaps], ignore_index=True)
df = df[['glucose']].dropna()

# ===============================================================
# NORMALIZATION
# ===============================================================
scaler = MinMaxScaler()
df['glucose_norm'] = scaler.fit_transform(df[['glucose']])

# ===============================================================
# CREATE DATASET
# ===============================================================
def create_dataset(data, seq_len=12):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 12

dataset = df['glucose_norm'].values
X, y = create_dataset(dataset, seq_len=SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ===============================================================
# BUILD MODEL
# ===============================================================
def build_model():
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

model = build_model()

# ===============================================================
# TRAIN MODEL
# ===============================================================
es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0, callbacks=[es])

# ===============================================================
# MODEL PERFORMANCE
# ===============================================================
pred_test = model.predict(X_test, verbose=0)
pred_test_real = scaler.inverse_transform(pred_test)
y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))

rmse = np.sqrt(mean_squared_error(y_test_real, pred_test_real))
mae  = mean_absolute_error(y_test_real, pred_test_real)
mape = np.mean(np.abs((y_test_real - pred_test_real) / y_test_real)) * 100

print("\n=========== MODEL PERFORMANCE ===========")
print(f"RMSE: {rmse:.3f} mg/dL")
print(f"MAE : {mae:.3f} mg/dL")
print(f"MAPE: {mape:.2f}%")
print("=========================================\n")

# ===============================================================
# MULTI-STEP FORECAST FUNCTION
# ===============================================================
def multi_step_forecast(model, last_seq, steps=12):
    seq = last_seq.copy().reshape(1, SEQ_LEN, 1)
    preds = []

    for _ in range(steps):
        pred = model.predict(seq, verbose=0)
        preds.append(pred[0][0])
        seq = np.concatenate([seq[:,1:,:], pred.reshape(1,1,1)], axis=1)

    return np.array(preds)

# Last window
last_seq = X_test[-1]

pred_15 = multi_step_forecast(model, last_seq, 3)
pred_30 = multi_step_forecast(model, last_seq, 6)
pred_60 = multi_step_forecast(model, last_seq, 12)

p15_value = scaler.inverse_transform(pred_15.reshape(-1,1))[-1][0]
p30_value = scaler.inverse_transform(pred_30.reshape(-1,1))[-1][0]
p60_value = scaler.inverse_transform(pred_60.reshape(-1,1))[-1][0]

print("15 min prediction :", p15_value, "mg/dL")
print("30 min prediction :", p30_value, "mg/dL")
print("60 min prediction :", p60_value, "mg/dL\n")

# ===============================================================
# SAFETY ALERT + RECOMMENDATIONS
# ===============================================================
def glucose_status(value):
    value = float(value)
    if value < 70:
        return ("🆘 SEVERE LOW GLUCOSE", "Eat 15–20g fast carbs. Rest immediately.")
    elif value < 90:
        return ("⚠️ LOW GLUCOSE WARNING", "Have a snack, monitor in 20 min.")
    elif value <= 160:
        return ("✅ NORMAL RANGE", "Great! Stay hydrated.")
    elif value <= 220:
        return ("⚠️ HIGH GLUCOSE WARNING", "Light walk 10 min. Avoid sugar.")
    else:
        return ("🆘 VERY HIGH GLUCOSE", "Stop activity, drink water, seek help.")

print("\n================ HEALTH ALERT SYSTEM ================\n")
for label, val in zip(["15-Min", "30-Min", "60-Min"], [p15_value, p30_value, p60_value]):
    status, advice = glucose_status(val)
    print(f"{label}: {val:.2f} mg/dL")
    print("Status:", status)
    print("Advice:", advice, "\n")

# ===============================================================
# RISK ANALYSIS SYSTEM (NEW)
# ===============================================================
def risk_level(glucose_values):
    p15, p30, p60 = glucose_values
    
    # Trend
    if p60 > p30 > p15:
        trend = "Rising 🔼"
    elif p60 < p30 < p15:
        trend = "Falling 🔽"
    else:
        trend = "Stable ➖"

    # Spike risk
    dif1 = abs(p30 - p15)
    dif2 = abs(p60 - p30)

    if dif1 > 25 or dif2 > 25:
        spike_risk = "HIGH ⚠️"
    elif dif1 > 15 or dif2 > 15:
        spike_risk = "MEDIUM ⚠"
    else:
        spike_risk = "LOW ✔"

    # Short-term risk classification
    def classify(v):
        if v < 70: return "Hypoglycemia Risk 🆘"
        if v < 90: return "Low Warning ⚠️"
        if v > 240: return "Severe Hyperglycemia 🆘"
        if v > 180: return "High Warning ⚠️"
        return "Stable ✔"

    return trend, spike_risk, classify(p15), classify(p30), classify(p60)

trend, spike_risk, risk15, risk30, risk60 = risk_level([p15_value, p30_value, p60_value])

print("\n================ RISK ANALYSIS ================\n")
print("Trend:", trend)
print("Spike Risk:", spike_risk)
print("15-min Risk:", risk15)
print("30-min Risk:", risk30)
print("60-min Risk:", risk60)
print("\n================================================\n")

# ===============================================================
# VISUALIZATION
# ===============================================================
plt.figure(figsize=(10,5))
plt.plot(y_test_real, label="Actual")
plt.plot(pred_test_real, label="Predicted")
plt.title("Blood Glucose Prediction - LSTM")
plt.xlabel("Time")
plt.ylabel("Glucose (mg/dL)")
plt.legend()
plt.grid()
plt.show()

# ===============================================================
# INCREMENTAL LEARNING
# ===============================================================
def incremental_train(model, new_values):
    X_new, y_new = create_dataset(new_values, SEQ_LEN)
    X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))
    if len(X_new) > 5:
        model.fit(X_new, y_new, epochs=10, batch_size=16, verbose=0)
    return model

print("Incremental learning enabled.")