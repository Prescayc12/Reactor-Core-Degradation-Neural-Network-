from data import load_data, get_scaler
from train import run_kfold

if __name__ == "__main__":
    X, y = load_data("../rvid2.xlsx")
    run_kfold(X, y, get_scaler)
