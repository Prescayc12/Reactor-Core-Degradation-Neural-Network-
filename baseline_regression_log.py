import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Load & Clean ─────────────────────────────────────────────────────────────
df = pd.read_excel("rvid2.xlsx", sheet_name="RVID2")

FEATURES = ["%Cu", "%Ni", "%P", "%S", "f at EOL 1/4T", "RTndt (u) [Initial RTndt]"]
TARGET   = "ΔRTndt"

df = df[FEATURES + [TARGET]].dropna()

df["log_fluence"] = np.log(df["f at EOL 1/4T"])
FEATURES = ["%Cu", "%Ni", "%P", "%S", "log_fluence", "RTndt (u) [Initial RTndt]"]

X = df[FEATURES].values
y = df[TARGET].values

# ── Split ─────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ── Fit ───────────────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"RMSE : {rmse:.2f} °F")
print(f"MAE  : {mae:.2f} °F")
print(f"R²   : {r2:.4f}")

print("\nCoefficients:")
for feat, coef in zip(FEATURES, model.coef_):
    print(f"  {feat:<35} {coef:+.4f}")
print(f"  {'Intercept':<35} {model.intercept_:+.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4, color="steelblue")

lims = [min(y_test.min(), y_pred.min()) - 10,
        max(y_test.max(), y_pred.max()) + 10]
ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")

ax.set_xlabel("Measured ΔRTndt (°F)")
ax.set_ylabel("Predicted ΔRTndt (°F)")
ax.set_title(f"Baseline Linear Regression — Predicted vs. Measured\nRMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}")
ax.legend()
plt.tight_layout()
plt.savefig("baseline_predicted_vs_measured_log.png", dpi=150)
plt.show()
