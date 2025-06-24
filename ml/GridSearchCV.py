import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Suppose you have already created df_ml (with features & labels) and saved as CSV
df_ml = pd.read_csv('data/df_ml.csv')

features = ["RSI_14", "DMA_20", "DMA_50", "DMA_100", "SUPPORT_20", "RESIST_20", "PE", "PB"]
X = df_ml[features]
y = df_ml["ml_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced']
}



clf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=3,                # 3-fold cross-validation
    n_jobs=-1,           # Use all CPUs
    verbose=2,
    scoring='f1_macro'   # Use macro F1 to balance classes
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Predict with best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
