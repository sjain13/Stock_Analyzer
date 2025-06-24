import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


df_ml = pd.read_csv('data/df_ml.csv')

# Example features - add/remove as needed:
features = [
    "RSI_14", "DMA_20", "DMA_50", "DMA_100", "SUPPORT_20", "RESIST_20", "PE", "PB"
]
X = df_ml[features].values
y = df_ml["ml_label"].values

# Encode Target Variable
label_enc = LabelEncoder()
y_int = label_enc.fit_transform(y)
y_cat = to_categorical(y_int)  # for multiclass (BUY/SELL/HOLD)


X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


model = Sequential()
model.add(LSTM(32, input_shape=(1, X_train_lstm.shape[2]), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(y_cat.shape[1], activation='softmax'))  # num_classes = y_cat.shape[1]
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train_lstm, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)

score = model.evaluate(X_test_lstm, y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# For classification report
from sklearn.metrics import classification_report
y_pred_probs = model.predict(X_test_lstm)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=label_enc.classes_))
