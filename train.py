"""
train.py  —  run this ONCE to train and save the model.
Output: model.pkl, scaler.pkl, encoder.pkl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib, os

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "diamonds.csv"          # put your CSV next to this file
MODEL_DIR  = "model"                 # folder where .pkl files are saved
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 1. load & clean ───────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 0", "x", "y", "z"], errors="ignore")

# ── 2. ordinal encoding ───────────────────────────────────────────────────────
cut_order     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_order   = ["J", "I", "H", "G", "F", "E", "D"]
clarity_order = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
df[["cut", "color", "clarity"]] = encoder.fit_transform(df[["cut", "color", "clarity"]])

# ── 3. split ──────────────────────────────────────────────────────────────────
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. scale  (kept for completeness — XGBoost doesn't need it,
#              but saving scaler lets you add linear models later) ─────────────
scaler = StandardScaler()
scaler.fit(X_train)          # fit only on train — no leakage

# ── 5. train xgboost ─────────────────────────────────────────────────────────
xgb = XGBRegressor(
    n_estimators     = 300,
    learning_rate    = 0.1,
    max_depth        = 6,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42,
    n_jobs           = -1,
)
xgb.fit(X_train, y_train)

# ── 6. evaluate ───────────────────────────────────────────────────────────────
preds = xgb.predict(X_test)
print(f"MSE:      {mean_squared_error(y_test, preds):,.0f}")
print(f"R2 Score: {r2_score(y_test, preds):.4f}")

# ── 7. save artifacts ─────────────────────────────────────────────────────────
joblib.dump(xgb,     f"{MODEL_DIR}/model.pkl")
joblib.dump(scaler,  f"{MODEL_DIR}/scaler.pkl")
joblib.dump(encoder, f"{MODEL_DIR}/encoder.pkl")

print(f"\nSaved → {MODEL_DIR}/model.pkl")
print(f"Saved → {MODEL_DIR}/scaler.pkl")
print(f"Saved → {MODEL_DIR}/encoder.pkl")
