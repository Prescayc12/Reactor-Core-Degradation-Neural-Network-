import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Update features to include your new engineered terms
FEATURES = [
    "%Cu", "%Ni", "%P", "%S", "log_fluence", "RTndt (u) [Initial RTndt]",
    "Cu_LogFluence", "Ni_LogFluence", "Cu_Ni"
]
TARGET = "ΔRTndt"

def load_data(filepath="rvid2.xlsx"):
    df = pd.read_excel(filepath, sheet_name="RVID2")
    df.columns = df.columns.str.strip()

    # Define base columns needed for calculation
    raw_cols = ["%Cu", "%Ni", "%P", "%S", "f at EOL 1/4T", "RTndt (u) [Initial RTndt]", TARGET]
    df = df[raw_cols].dropna()

    # Feature Engineering
    df["log_fluence"] = np.log(df["f at EOL 1/4T"])
    
    # Interaction Terms
    df["Cu_LogFluence"] = df["%Cu"] * df["log_fluence"]
    df["Ni_LogFluence"] = df["%Ni"] * df["log_fluence"]
    df["Cu_Ni"] = df["%Cu"] * df["%Ni"]

    X = df[FEATURES].values
    y = df[TARGET].values

    return X, y

def get_scaler():
    return StandardScaler()
