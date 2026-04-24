import optuna
import time
from data import load_data, get_scaler, ALL_FEATURES
from train import run_kfold

TIME_LIMIT_SECONDS = 28800  # 8 hours

def objective(trial, X_full, y):
    # ── Feature Selection ─────────────────────────────────────────────────────
    # %Cu is always included, at least one other must be selected
    optional_features = [f for f in ALL_FEATURES if f != "%Cu"]
    selected = ["%Cu"]
    for feat in optional_features:
        if trial.suggest_categorical(f"use_{feat}", [True, False]):
            selected.append(feat)

    # Enforce minimum 2 features
    if len(selected) < 2:
        selected.append(trial.suggest_categorical("forced_feature", optional_features))

    # Load data with selected features
    from data import load_data as _load
    X, y = _load(selected_features=selected)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    if optimizer_name == "sgd":
        lr       = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
    else:
        lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        momentum = 0.9

    config = {
        # Features
        "selected_features": selected,
        # Hyperparameters
        "epochs":            trial.suggest_int("epochs", 500, 3000),
        "lr":                lr,
        "weight_decay":      trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout_rate":      trial.suggest_float("dropout_rate", 0.0, 0.5),
        "optimizer":         optimizer_name,
        "momentum":          momentum,
        "scheduler":         trial.suggest_categorical("scheduler", [None, "cosine", "step", "exponential"]),
        "init":              trial.suggest_categorical("init", ["kaiming", "xavier", "normal"]),
        # Architecture
        "n_layers":          trial.suggest_int("n_layers", 2, 4),
        "first_layer_size":  trial.suggest_categorical("first_layer_size", [32, 64, 128, 256]),
        "activation":        trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu"]),
        "batch_norm":        trial.suggest_categorical("batch_norm", [True, False]),
    }

    rmse = run_kfold(X, y, get_scaler, config, verbose=False)
    return rmse

def main():
    X_full, y = load_data()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="rvid_optimization_final",
        storage="sqlite:///optuna_final.db",
        load_if_exists=True,
        direction="minimize"
    )

    start = time.time()
    print(f"Starting final Optuna search — 8 hour limit, unlimited trials")
    print(f"Searching: features, architecture, hyperparameters\n")

    study.optimize(
        lambda trial: objective(trial, X_full, y),
        n_trials=None,
        timeout=TIME_LIMIT_SECONDS,
        catch=(Exception,),
        show_progress_bar=True
    )

    elapsed = time.time() - start
    print(f"\nSearch complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Trials completed: {len(study.trials)}")
    failed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)
    print(f"Trials failed: {failed}")
    print(f"\nBest RMSE: {study.best_value:.4f}°F")
    print(f"\nBest config:")
    for k, v in study.best_params.items():
        print(f"  {k:<30} {v}")

if __name__ == "__main__":
    main()
