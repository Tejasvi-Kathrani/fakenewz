# %% [markdown]
# Fake News Classifier (TF-IDF + LogisticRegression)
# Paste cells into a Jupyter notebook or run as a script (cells separated by # %%)

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# %%
# CONFIG: adjust paths and column names as needed
DATA_PATH = "fake_or_real_news.csv"   # replace with your dataset path
TEXT_COL = "text"
LABEL_COL = "label"
RANDOM_STATE = 42

# %%
# Load and inspect data
df = pd.read_csv(DATA_PATH)
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
try:
    display(df.head())
except Exception:
    print(df.head().to_string())

# %%
# Basic cleaning
df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].astype(str)
# Map labels to binary integers if needed
if df[LABEL_COL].dtype == object:
    label_values = df[LABEL_COL].unique().tolist()
    if set(["FAKE","REAL"]).issubset(set(label_values)) or set(["fake","real"]).issubset(set([v.lower() for v in label_values])):
        df["label_num"] = df[LABEL_COL].str.upper().map({"FAKE": 0, "REAL": 1})
    else:
        # fallback: encode with factorize
        df["label_num"], uniques = pd.factorize(df[LABEL_COL])
else:
    df["label_num"] = df[LABEL_COL].astype(int)

print("Label distribution:")
print(df["label_num"].value_counts())

# %%
# Train/test split
X = df[TEXT_COL].values
y = df["label_num"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train / Test sizes:", len(X_train), len(X_test))

# %%
# Build pipeline
pipeline = make_pipeline(
    TfidfVectorizer(
        max_features=20000,
        ngram_range=(1,2),
        stop_words="english"
    ),
    LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE
    )
)

# %%
# Train
pipeline.fit(X_train, y_train)

# %%
# Evaluate
y_pred = pipeline.predict(X_test)
print(metrics.classification_report(y_test, y_pred, digits=4))

cm = metrics.confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["FAKE","REAL"], yticklabels=["FAKE","REAL"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# %%
# Show ROC AUC and PR AUC (if you want probabilities)
if hasattr(pipeline, "predict_proba"):
    y_proba = pipeline.predict_proba(X_test)[:,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    auc = metrics.roc_auc_score(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# %%
# Save model
MODEL_PATH = "fake_news_pipeline.joblib"
joblib.dump(pipeline, MODEL_PATH)
print("Saved model to", MODEL_PATH)

# %%
# Helper predict function
def predict_text(texts, model_path=MODEL_PATH):
    """
    texts: str or list[str]
    returns DataFrame with text, predicted_label, and prob (when available)
    """
    if isinstance(texts, str):
        texts = [texts]
    pipe = joblib.load(model_path)
    preds = pipe.predict(texts)
    out = {"text": texts, "pred": preds}
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(texts)[:,1]
        out["prob_real"] = proba
    return pd.DataFrame(out)

# Example usage:
# print(predict_text("The earth is flat and scientists are hiding it."))
