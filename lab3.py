from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

DATA_PATH = Path(r"C:\Users\Полина\Desktop\УлГТУ магистратура\ИИ\laba 3\Extended_Employee_Performance_and_Productivity_Data.csv")
TARGET = "Performance_Score"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TFIDF_MAX_FEATURES = 1000
TFIDF_MIN_DF = 10

OUT_DIR = DATA_PATH.parent / "regression_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig").dropna(axis=1, how='all').drop_duplicates()
y = df[TARGET]
X = df.drop(columns=[TARGET])

text_candidates = [c for c in ["Department","Job_Title","Role","Team","Seniority"] if c in X.columns]
if text_candidates:
    X["TextFeature"] = X[text_candidates].astype(str).agg(" ".join, axis=1)
else:
    cat_all = X.select_dtypes(include=["object","category"]).columns.tolist()
    X["TextFeature"] = X[cat_all].astype(str).agg(" ".join, axis=1) if cat_all else ""

n = len(X)
high_card = [c for c in X.select_dtypes(include=["object","category"]).columns
             if X[c].nunique(dropna=False) / n > 0.9 and c != "TextFeature"]
id_like_num = [c for c in X.select_dtypes(include=[np.number]).columns
               if X[c].nunique(dropna=False) / n > 0.9]

X = X.drop(columns=high_card + id_like_num)

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=["object","category"]).columns.tolist()
if "TextFeature" in categorical_features:
    categorical_features.remove("TextFeature")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF)

pre_tabular = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

pre_text = ColumnTransformer([
    ("text", tfidf, "TextFeature"),
])

pre_combined = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
    ("text", tfidf, "TextFeature"),
])

configs = {"tabular": pre_tabular, "text": pre_text, "combined": pre_combined}

rows = []
for name, pre in configs.items():
    pipe = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - t0
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    rows.append([name, r2, mae, rmse, fit_time])
    pd.DataFrame(rows, columns=["features","R2","MAE","RMSE","fit_time"]).to_csv(OUT_DIR/"results.csv", index=False)

    residuals = y_test.values - y_pred
    plt.figure(); plt.scatter(y_test, y_pred, s=10); lims=[min(y_test.min(),y_pred.min()),max(y_test.max(),y_pred.max())]
    plt.plot(lims, lims); plt.savefig(OUT_DIR/f"{name}_true_vs_pred.png"); plt.close()

    plt.figure(); plt.hist(residuals, bins=30); plt.savefig(OUT_DIR/f"{name}_residuals_hist.png"); plt.close()

    plt.figure(); plt.scatter(y_pred, residuals, s=10); plt.axhline(0); plt.savefig(OUT_DIR/f"{name}_residuals_vs_pred.png"); plt.close()

print("Готово")
