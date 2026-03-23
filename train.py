import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import os

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("Data/processed/features.csv")
df.columns = df.columns.str.strip()

# convert bool → int
df = df.apply(lambda col: col.astype(int) if col.dtype == "bool" else col)

print("DATA SHAPE:", df.shape)

# =============================
# TARGETS
# =============================
target_ext = "extended_icu"
target_readmit = "readmit_30"
target_los = "icu_los_hours"

# =============================
# FEATURES
# =============================
X = df.drop([target_ext, target_readmit, target_los], axis=1)

# save feature columns
feature_cols = X.columns
joblib.dump(feature_cols, "models/feature_columns.pkl")

# =============================
# SCALER
# =============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "models/scaler.pkl")

# =============================
# TRAIN-TEST SPLIT
# =============================
X_train, X_test, y_ext_train, y_ext_test = train_test_split(
    X_scaled, df[target_ext], test_size=0.2, random_state=42
)

_, _, y_read_train, y_read_test = train_test_split(
    X_scaled, df[target_readmit], test_size=0.2, random_state=42
)

_, _, y_los_train, y_los_test = train_test_split(
    X_scaled, df[target_los], test_size=0.2, random_state=42
)

# =============================
# CONFUSION MATRIX FUNCTION
# =============================
def save_conf_matrix(y_true, y_pred, name):
    # create folder if not exists
    os.makedirs("Confusion matrix", exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format='d')

    plt.title(f"{name} Confusion Matrix")

    # save inside folder
    path = f"Confusion matrix/{name}_confusion_matrix.png"
    plt.savefig(path)

    plt.close()

    print(f" Saved: {path}")


# =============================
# 1️⃣ EXTENDED ICU MODEL
# =============================
print("\n TRAINING EXTENDED ICU MODEL...")

model_ext = RandomForestClassifier(n_estimators=300, random_state=42)
model_ext.fit(X_train, y_ext_train)

pred_ext = model_ext.predict(X_test)
prob_ext = model_ext.predict_proba(X_test)[:, 1]

print("\nEXTENDED ICU REPORT")
print(classification_report(y_ext_test, pred_ext))
print("ROC-AUC:", roc_auc_score(y_ext_test, prob_ext))

# SAVE CONFUSION MATRIX
save_conf_matrix(y_ext_test, pred_ext, "extended_icu")

joblib.dump(model_ext, "models/model_extended_icu.pkl")


# =============================
# 2️⃣ READMISSION MODEL
# =============================
print("\n TRAINING READMISSION MODEL...")

model_read = RandomForestClassifier(n_estimators=300, random_state=42)
model_read.fit(X_train, y_read_train)

pred_read = model_read.predict(X_test)
prob_read = model_read.predict_proba(X_test)[:, 1]

print("\nREADMISSION REPORT")
print(classification_report(y_read_test, pred_read))
print("ROC-AUC:", roc_auc_score(y_read_test, prob_read))

# SAVE CONFUSION MATRIX
save_conf_matrix(y_read_test, pred_read, "readmission")

joblib.dump(model_read, "models/model_readmission.pkl")


# =============================
# 3️⃣ ICU LOS MODEL (REGRESSION)
# =============================
print("\n TRAINING ICU LOS MODEL...")

model_los = RandomForestRegressor(n_estimators=300, random_state=42)
model_los.fit(X_train, y_los_train)

pred_los = model_los.predict(X_test)

print("\nICU LOS RESULTS")
print("MAE:", mean_absolute_error(y_los_test, pred_los))
print("R2 Score:", r2_score(y_los_test, pred_los))

joblib.dump(model_los, "models/model_icu_los.pkl")


# =============================
# DONE
# =============================
print("\n ALL MODELS TRAINED & SAVED SUCCESSFULLY")