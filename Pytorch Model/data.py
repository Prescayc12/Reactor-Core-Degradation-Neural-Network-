import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

ALL_FEATURES = ["%Cu", "%Ni", "%P", "%S", "log_fluence", "RTndt (u) [Initial RTndt]", "Cu_LogFluence"]
TARGET = "ΔRTndt"

def load_data(filepath="../rvid2.xlsx", selected_features=None):
    df = pd.read_excel(filepath, sheet_name="RVID2")
    df.columns = df.columns.str.strip()

    raw_cols = ["%Cu", "%Ni", "%P", "%S", "f at EOL 1/4T", "RTndt (u) [Initial RTndt]", TARGET]
    df = df[raw_cols].dropna()

    df["log_fluence"]    = np.log(df["f at EOL 1/4T"])
    df["Cu_LogFluence"]  = df["%Cu"] * df["log_fluence"]

    if selected_features is None:
        selected_features = ALL_FEATURES

    X = df[selected_features].values
    y = df[TARGET].values

    return X, y

def get_scaler():
    return StandardScaler()
