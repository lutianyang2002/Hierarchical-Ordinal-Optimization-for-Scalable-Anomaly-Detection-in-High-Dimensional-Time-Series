import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import random

np.random.seed(42)
random.seed(42)

file_pairs = [
    ("result1_output.csv", "result1_fine1_anomaly_scores.csv"),
    ("result2_output.csv", "result2_fine1_anomaly_scores.csv"),
    ("result3_output.csv", "result3_fine1_anomaly_scores.csv"),
    ("result1_output.csv", "result1_fine2_anomaly_scores.csv"),
    ("result2_output.csv", "result2_fine2_anomaly_scores.csv"),
    ("result3_output.csv", "result3_fine2_anomaly_scores.csv"),
    ("result1_output.csv", "result1_fine3_anomaly_scores.csv"),
    ("result2_output.csv", "result2_fine3_anomaly_scores.csv"),
    ("result3_output.csv", "result3_fine3_anomaly_scores.csv"),
]

# Fit surrogate model and compute SHAP
def fit_and_explain(coarse_csv: str, fine_csv: str):
    # read coarse features (contains: Index + real features + Result)
    X_df = pd.read_csv(coarse_csv)
    # read fine targets (Index + Anomaly_Score)
    y_df = pd.read_csv(fine_csv)

    # merge by Index to align
    merged = pd.merge(
        X_df, y_df[["Index", "Anomaly_Score"]],
        on="Index", how="inner", validate="one_to_one"
    ).sort_values("Index").reset_index(drop=True)

    # drop columns that are NOT features
    X = merged.drop(columns=["Index", "Anomaly_Score", "Result"], errors="ignore")
    y = merged["Anomaly_Score"].values

    # standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # surrogate model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)

    # SHAP calculation
    explainer = shap.Explainer(model, X_scaled, feature_names=X.columns)
    shap_values = explainer(X_scaled)

    return shap_values, X_scaled, X.columns

# Plot the SHAP figure
fig, axes = plt.subplots(3, 3, figsize=(22, 18))
axes = axes.flatten()

for i, (coarse_file, fine_file) in enumerate(file_pairs):
    shap_values, X_scaled, feature_names = fit_and_explain(coarse_file, fine_file)

    plt.sca(axes[i])
    shap.summary_plot(
        shap_values,
        features=X_scaled,
        feature_names=feature_names,
        show=False,
        plot_size=(8, 6),
        max_display=10,
        color_bar=True
    )
    axes[i].set_title(
        fine_file.replace("_anomaly_scores.csv", ""),
        fontsize=11, fontweight="bold"
    )
    axes[i].tick_params(labelsize=8)
    for label in axes[i].get_yticklabels():
        label.set_fontsize(8)

plt.tight_layout()
plt.show()
