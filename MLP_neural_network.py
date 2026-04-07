import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
# Works with rvid2.xlsx or the extracted RVID2 csv
filename = 'rvid2.xlsx'
try:
    df = pd.read_excel(filename, sheet_name="RVID2")
except:
    # Fallback if file is named differently or only CSV is available
    df = pd.read_csv("rvid2.xlsx - RVID2.csv")

df.columns = df.columns.str.strip()

# 2. Map Columns (Dynamic search for headers)
def find_col(keys):
    for k in keys:
        for c in df.columns:
            if k.lower() in str(c).lower(): return c
    return None

mapping = {
    "Cu": find_col(["%Cu", "Cu"]),
    "Ni": find_col(["%Ni", "Ni"]),
    "P": find_col(["%P", "P"]),
    "S": find_col(["%S", "S"]),
    "Fluence": find_col(["f at EOL", "Fluence"]),
    "RT_init": find_col(["Initial RTndt", "RTndt(u)"]),
    "Target": find_col(["ΔRTndt", "Shift", "dRT"]) or df.columns[-1]
}

# 3. Preprocessing & Feature Engineering
df_model = df[list(mapping.values())].copy()
df_model.columns = mapping.keys()
df_model = df_model.apply(pd.to_numeric, errors='coerce').dropna()

# Natural log of Fluence as per project proposal
df_model["log_fluence"] = np.log(df_model["Fluence"])
FEATURES = ["Cu", "Ni", "P", "S", "log_fluence", "RT_init"]

X = df_model[FEATURES].values
y = df_model["Target"].values

# 4. K-Fold Cross-Validation (5 Folds)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results_lr, results_mlp = [], []
all_y_test, all_y_pred_lr, all_y_pred_mlp = [], [], []

print(f"{'Fold':<6} | {'LR RMSE':<10} | {'MLP RMSE':<10}")
print("-" * 35)

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    # Split
    X_train_f, X_test_f = X[train_idx], X[test_idx]
    y_train_f, y_test_f = y[train_idx], y[test_idx]
    
    # Scale (Fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_f)
    X_test_scaled = scaler.transform(X_test_f)
    
    # Train Baseline (Linear Regression)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_f)
    y_pred_lr = lr.predict(X_test_scaled)
    
    # Train Proposed Model (Neural Network)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', 
                       solver='adam', max_iter=1500, random_state=42)
    mlp.fit(X_train_scaled, y_train_f)
    y_pred_mlp = mlp.predict(X_test_scaled)
    
    # Store results
    rmse_lr = np.sqrt(mean_squared_error(y_test_f, y_pred_lr))
    rmse_mlp = np.sqrt(mean_squared_error(y_test_f, y_pred_mlp))
    results_lr.append(rmse_lr)
    results_mlp.append(rmse_mlp)
    
    # Track all predictions for final plot
    all_y_test.extend(y_test_f)
    all_y_pred_lr.extend(y_pred_lr)
    all_y_pred_mlp.extend(y_pred_mlp)
    
    print(f"#{fold+1:<5} | {rmse_lr:<10.2f} | {rmse_mlp:<10.2f}")

print("-" * 35)
print(f"AVERAGE| {np.mean(results_lr):<10.2f} | {np.mean(results_mlp):<10.2f}")

# 5. Visual Comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(all_y_test, all_y_pred_lr, alpha=0.4, color='steelblue')
plt.plot([0, 300], [0, 300], 'r--')
plt.title("Baseline: Linear Regression")
plt.xlabel("Measured ΔRTndt (°F)"); plt.ylabel("Predicted ΔRTndt (°F)")

plt.subplot(1, 2, 2)
plt.scatter(all_y_test, all_y_pred_mlp, alpha=0.4, color='seagreen')
plt.plot([0, 300], [0, 300], 'r--')
plt.title("Improved: MLP Neural Network")
plt.xlabel("Measured ΔRTndt (°F)"); plt.ylabel("Predicted ΔRTndt (°F)")

plt.tight_layout()
plt.show()
