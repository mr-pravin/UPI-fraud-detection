# train.py (sketch)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import optuna
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE

# load
df = pd.read_csv("data/processed/transactions.csv", parse_dates=["ts"])
X = df.drop(columns=["is_fraud", "txn_id", "ts"])
y = df["is_fraud"]

# feature pipeline
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

preproc = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
])

# model + pipeline
model = xgb.XGBClassifier(tree_method="hist", use_label_encoder=False, eval_metric="logloss")
pipe = Pipeline([("preproc", preproc), ("smote", SMOTE()), ("clf", model)])

# time-based CV
tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    params = {
        "clf__max_depth": trial.suggest_int("max_depth", 3, 10),
        "clf__learning_rate": trial.suggest_loguniform("lr", 1e-3, 1e-1),
        "clf__n_estimators": trial.suggest_int("n_estimators", 100, 1000)
    }
    pipe.set_params(**params)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict_proba(X.iloc[val_idx])[:,1]
        # compute PR-AUC or custom cost-based metric
        from sklearn.metrics import average_precision_score
        scores.append(average_precision_score(y.iloc[val_idx], preds))
    return np.mean(scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
best = study.best_params
pipe.set_params(**best)
pipe.fit(X, y)
joblib.dump(pipe, "models/upi_fraud_xgb.pkl")
