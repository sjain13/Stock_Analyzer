import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Suppose you have already created df_ml (with features & labels) and saved as CSV
df_ml = pd.read_csv('data/df_ml.csv')

features = ["RSI_14", "DMA_20", "DMA_50", "DMA_100", "SUPPORT_20", "RESIST_20", "PE", "PB"]
X = df_ml[features]
y = df_ml["ml_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model for later inference
joblib.dump(clf, 'data/stock_signal_rf_model.pkl')
