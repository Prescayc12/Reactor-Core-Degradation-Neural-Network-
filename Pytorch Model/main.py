import time
from data import load_data, get_scaler
from train import run_kfold

BEST_CONFIG = {
    "selected_features": ["%Cu", "%Ni", "%S", "log_fluence", "RTndt (u) [Initial RTndt]"],
    "epochs":            2546,
    "lr":                0.0016270329288117422,
    "weight_decay":      0.005785644921206395,
    "dropout_rate":      0.034005334601626844,
    "optimizer":         "adam",
    "momentum":          0.9,
    "scheduler":         "cosine",
    "init":              "kaiming",
    "n_layers":          3,
    "first_layer_size":  256,
    "activation":        "relu",
    "batch_norm":        False,
}

if __name__ == "__main__":
    X, y = load_data(selected_features=BEST_CONFIG["selected_features"])
    start = time.time()
    run_kfold(X, y, get_scaler, config=BEST_CONFIG)
    elapsed = time.time() - start
    print(f"\nSingle run time: {elapsed:.1f}s ({elapsed/60:.2f} min)")
