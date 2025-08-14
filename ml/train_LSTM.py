import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt



# ==============================
# CONFIGURATION
# ==============================
DATA_PATH = "data/df_ml_new.csv"
MODEL_PATH = "data/lstm_model.h5"
SCALER_PATH = "data/lstm_scaler.pkl"
LABEL_ENCODER_PATH = "data/lstm_label_encoder.pkl"
WINDOW = 20

# Features to use (including fundamental & technical)
features = [
    "close", "volume", "RSI_14", "DMA_20", "DMA_50", "DMA_100",
    "SUPPORT_20", "RESIST_20", "PE", "PB"
]

# ==============================
# FUNCTIONS
# ==============================

def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows.")
    return df

def clean_data(df, features, label='ml_label'):
    # Drop rows where any required feature or label is missing
    df_clean = df.dropna(subset=features + [label]).copy()
    # Fill any remaining NaN with forward/backward fill (if needed)
    df_clean[features] = df_clean[features].fillna(method='ffill').fillna(method='bfill')
    # Remove extreme outliers (1st, 99th percentiles) -- optional
    # for col in features:
    #     if pd.api.types.is_numeric_dtype(df_clean[col]):
    #         q1, q99 = df_clean[col].quantile([0.01, 0.99])
    #         df_clean = df_clean[(df_clean[col] >= q1) & (df_clean[col] <= q99)]
    return df_clean

def encode_labels(df, target_label='ml_label'):
    le = LabelEncoder()
    df['label'] = le.fit_transform(df[target_label])
    return df, le

def print_data_status(df, features, label='ml_label'):
    print("\n===== Data Status =====")
    print("Head:\n", df.head())
    print("\nDescribe:\n", df[features].describe())
    print("\nMissing values:\n", df[features + [label]].isnull().sum())
    print("\nLabel distribution:\n", df[label].value_counts())
    print(f"\nUnique instruments: {df['instrument_id'].nunique() if 'instrument_id' in df.columns else 'N/A'}")
    print(f"PE Ratio: min={df['PE'].min()}, max={df['PE'].max()}, mean={df['PE'].mean()}")
    print(f"PB Ratio: min={df['PB'].min()}, max={df['PB'].max()}, mean={df['PB'].mean()}")

def create_sequences(X, y, window=20):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

def build_lstm_model(input_shape, n_classes=3):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

# ==============================
# MAIN PIPELINE
# ==============================
df = load_data(DATA_PATH)
df_clean = clean_data(df, features)
df_clean, label_encoder = encode_labels(df_clean)
print_data_status(df_clean, features)

# Scale only the numeric features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# Sequence generation (LSTM needs 3D input: [samples, timesteps, features])
X_seq, y_seq = create_sequences(X_scaled, df_clean['label'].values, window=WINDOW)
print("Input shape:", X_seq.shape, "Target shape:", y_seq.shape)

# Train/test split (no shuffle for time series!)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Class weights (use compute_class_weight for balance if needed)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Model
input_shape = (X_train.shape[1], X_train.shape[2])
model = build_lstm_model(input_shape)
model.summary()

# Training
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    class_weight=class_weights_dict
)

plot_history(history)

# Evaluation
y_pred = model.predict(X_test).argmax(axis=1)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['SELL', 'HOLD', 'BUY']))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=['SELL', 'HOLD', 'BUY']).plot()

# Save all for inference
joblib.dump(scaler, SCALER_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)
model.save(MODEL_PATH)
print("LSTM model, scaler, and label encoder saved!")

