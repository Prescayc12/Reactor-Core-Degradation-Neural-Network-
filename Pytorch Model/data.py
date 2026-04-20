import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Focusing only on the high-impact drivers
FEATURES = [
    "%Cu", 
    "%Ni", 
    "log_fluence", 
    "RTndt (u) [Initial RTndt]",
    "Cu_LogFluence"
]
TARGET = "ΔRTndt"

def load_data(filepath="rvid2.xlsx"):
    # Load and strip columns
    df = pd.read_excel(filepath, sheet_name="RVID2")
    df.columns = df.columns.str.strip()

    # Only pull the core columns needed
    raw_cols = ["%Cu", "%Ni", "f at EOL 1/4T", "RTndt (u) [Initial RTndt]", TARGET]
    df = df[raw_cols].dropna()

    # Log-transform fluence (Standard practice for radiation damage)
    df["log_fluence"] = np.log(df["f at EOL 1/4T"])
    
    # Single interaction term: Copper's sensitivity to radiation
    df["Cu_LogFluence"] = df["%Cu"] * df["log_fluence"]

    X = df[FEATURES].values
    y = df[TARGET].values

    return X, y

def get_scaler():
    # Keep the scaler consistent
    return StandardScaler()
