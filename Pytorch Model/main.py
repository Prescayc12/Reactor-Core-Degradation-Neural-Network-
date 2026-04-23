import time
from data import load_data, get_scaler
from train import run_kfold

BEST_CONFIG = {
    "epochs":       2339,
    "lr":           0.00023035821370080836,
    "weight_decay": 0.0011243156272627932,
    "dropout_rate": 0.0500993978361025,
    "optimizer":    "sgd",
    "momentum":     0.9,
    "scheduler":    "cosine",
    "init":         "kaiming",
}

if __name__ == "__main__":
    X, y = load_data("../rvid2.xlsx")
    start = time.time()
    run_kfold(X, y, get_scaler, config=BEST_CONFIG)
    elapsed = time.time() - start
    print(f"\nSingle run time: {elapsed:.1f}s ({elapsed/60:.2f} min)")
