import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import MLPRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CONFIG = {
    "epochs": 1500,
    "lr": 0.001,
    "weight_decay": 0.0,
    "dropout_rate": 0.0,
    "optimizer": "adam",
    "scheduler": None,
    "init": "kaiming",
    "n_layers": 2,
    "first_layer_size": 100,
    "activation": "relu",
    "batch_norm": False,
}

N_SPLITS = 5

def init_weights(model, method):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if method == "kaiming":
                nn.init.kaiming_uniform_(m.weight)
            elif method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method == "normal":
                nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.zeros_(m.bias)

def build_optimizer(model, config):
    opt = config.get("optimizer", "adam").lower()
    lr = config["lr"]
    wd = config.get("weight_decay", 0.0)
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == "sgd":
        momentum = config.get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif opt == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")

def build_scheduler(optimizer, config, epochs):
    sched = config.get("scheduler", None)
    if sched is None:
        return None
    elif sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.5)
    elif sched == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    else:
        raise ValueError(f"Unknown scheduler: {sched}")

def train_fold(model, X_train, y_train, optimizer, criterion, scheduler, epochs):
    model.train()
    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def eval_fold(model, X_test):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds = model(X_t).cpu().numpy()
    if np.any(np.isnan(preds)):
        raise ValueError("Model produced NaN predictions — trial skipped.")
    return preds

def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    std_dev = np.std(y_true - y_pred)
    return rmse, mae, r2, std_dev

def plot_fold(y_test, y_pred, fold, rmse, r2):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4, color="seagreen")
    lims = [min(y_test.min(), y_pred.min()) - 10, max(y_test.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Measured ΔRTndt (°F)")
    ax.set_ylabel("Predicted ΔRTndt (°F)")
    ax.set_title(f"Fold {fold} — RMSE={rmse:.2f}  R²={r2:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"fold_{fold}_predicted_vs_measured.png", dpi=150)
    plt.close()

def plot_aggregated(all_y_test, all_y_pred, rmse, mae, r2, std_dev):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_y_test, all_y_pred, alpha=0.5, edgecolors="k", linewidths=0.4, color="seagreen")
    lims = [min(all_y_test.min(), all_y_pred.min()) - 10, max(all_y_test.max(), all_y_pred.max()) + 10]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Measured ΔRTndt (°F)")
    ax.set_ylabel("Predicted ΔRTndt (°F)")
    ax.set_title(f"PyTorch MLP — Aggregated\nRMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  Std Dev={std_dev:.2f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("pytorch_mlp_aggregated.png", dpi=150)
    plt.close()

def run_kfold(X, y, scaler_fn, config=None, verbose=True):
    if config is None:
        config = DEFAULT_CONFIG

    epochs = config.get("epochs", 1500)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    criterion = nn.MSELoss().to(DEVICE)

    all_y_test  = []
    all_y_pred  = []
    fold_results = []

    if verbose:
        print(f"Running on: {DEVICE}")
        print(f"{'Fold':<6} | {'RMSE':<10} | {'MAE':<10} | {'R²':<10} | {'Std Dev':<10}")
        print("-" * 55)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler  = scaler_fn()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = MLPRegressor(input_size=X_train.shape[1], config=config).to(DEVICE)
        init_weights(model, config.get("init", "kaiming"))
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config, epochs)

        train_fold(model, X_train, y_train, optimizer, criterion, scheduler, epochs)
        y_pred = eval_fold(model, X_test)

        rmse, mae, r2, std_dev = compute_metrics(y_test, y_pred)
        fold_results.append((rmse, mae, r2, std_dev))

        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        if verbose:
            plot_fold(y_test, y_pred, fold, rmse, r2)
            print(f"#{fold:<5} | {rmse:<10.2f} | {mae:<10.2f} | {r2:<10.4f} | {std_dev:<10.2f}")

    avg = np.mean(fold_results, axis=0)

    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    rmse, mae, r2, std_dev = compute_metrics(all_y_test, all_y_pred)

    if verbose:
        print("-" * 55)
        print(f"{'AVG':<6} | {avg[0]:<10.2f} | {avg[1]:<10.2f} | {avg[2]:<10.4f} | {avg[3]:<10.2f}")
        plot_aggregated(all_y_test, all_y_pred, rmse, mae, r2, std_dev)
        print(f"\nAggregated across all folds:")
        print(f"  RMSE:    {rmse:.2f}°F")
        print(f"  MAE:     {mae:.2f}°F")
        print(f"  R²:      {r2:.4f}")
        print(f"  Std Dev: {std_dev:.2f}°F")
        print(f"\nRegulatory targets: Welds ≤28°F | Base Metal ≤17°F")

    return rmse
