import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Suppose you have already created df_ml (with features & labels) and saved as CSV
df_ml = pd.read_csv('data/df_ml.csv')

features = ["RSI_14", "DMA_20", "DMA_50", "DMA_100", "SUPPORT_20", "RESIST_20", "PE", "PB"]
X = df_ml[features]
y = df_ml["ml_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', None]
}

clf = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(
    estimator=clf,
    param_distributions=param_dist,
    n_iter=20,         # Number of random combos to try
    cv=3,
    n_jobs=-1,
    verbose=2,
    scoring='f1_macro',
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best parameters:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
