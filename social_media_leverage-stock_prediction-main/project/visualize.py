import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("predictions.csv")

# Must contain:
# date, target_1d, predicted_up, prob_up, confidence_level

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

y_true = df["target_1d"]
y_pred = df["predicted_up"]
y_prob = df["prob_up"]


# 2. Cumulative Profit Curve (Equity Curve)
df["strategy_return"] = df["return_1d"] * df["predicted_up"]
df["cumulative_profit"] = df["strategy_return"].cumsum()

plt.figure(figsize=(10,4))
plt.plot(df["date"], df["cumulative_profit"])
plt.title("Cumulative Profit Curve (Equity Curve)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid()
plt.tight_layout()
plt.show()

# 4. Calibration Curve
bins = np.linspace(0, 1, 11)
df["prob_bin"] = np.digitize(df["prob_up"], bins)
calibration = df.groupby("prob_bin").agg(
    avg_prob=("prob_up", "mean"),
    true_freq=("target_1d", "mean")
).dropna()

plt.figure(figsize=(6,6))
plt.plot(calibration["avg_prob"], calibration["true_freq"], marker="o")
plt.plot([0,1],[0,1], "--")
plt.title("Calibration Curve")
plt.xlabel("Predicted Probability")
plt.ylabel("Observed Frequency")
plt.grid()
plt.tight_layout()
plt.show()

# 5. Gains Chart / Lift Curve
df_sorted = df.sort_values("prob_up", ascending=False)
df_sorted["cumulative_positive"] = df_sorted["target_1d"].cumsum()
total_positives = df["target_1d"].sum()

df_sorted["gain"] = df_sorted["cumulative_positive"] / total_positives

plt.figure(figsize=(10,4))
plt.plot(np.arange(len(df_sorted)) / len(df_sorted), df_sorted["gain"])
plt.title("Gains Chart")
plt.xlabel("Proportion of Dataset (sorted by prob_up)")
plt.ylabel("Cumulative True Positive Rate")
plt.grid()
plt.tight_layout()
plt.show()
