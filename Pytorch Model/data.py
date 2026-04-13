import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FEATURES = ["%Cu", "%Ni", "%P", "%S", "log_fluence", "RTndt (u) [Initial RTndt]"]
TARGET = "ΔRTndt"

def load_data(filepath="rvid2.xlsx"):
    df = pd.read_excel(filepath, sheet_name="RVID2")
    df.columns = df.columns.str.strip()

    raw_features = ["%Cu", "%Ni", "%P", "%S", "f at EOL 1/4T", "RTndt (u) [Initial RTndt]"]
    df = df[raw_features + [TARGET]].dropna()

    df["log_fluence"] = np.log(df["f at EOL 1/4T"])

    X = df[FEATURES].values
    y = df[TARGET].values

    return X, y

def get_scaler():
    return StandardScaler()
