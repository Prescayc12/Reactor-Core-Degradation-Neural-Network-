import optuna
import time
from data import load_data, get_scaler
from train import run_kfold

N_TRIALS = 235
TIME_LIMIT_SECONDS = 7200  # 2 hours

def objective(trial, X, y):
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])

    if optimizer_name == "sgd":
        lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
    else:
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        momentum = 0.9  # unused for adam/rmsprop

    config = {
        "epochs":       trial.suggest_int("epochs", 500, 3000),
        "lr":           lr,
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
        "optimizer":    optimizer_name,
        "momentum":     momentum,
        "scheduler":    trial.suggest_categorical("scheduler", [None, "cosine", "step", "exponential"]),
        "init":         trial.suggest_categorical("init", ["kaiming", "xavier", "normal"]),
    }
    rmse = run_kfold(X, y, get_scaler, config)
    return rmse

def main():
    X, y = load_data("../rvid2.xlsx")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="rvid_optimization",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
        direction="minimize"
    )

    start = time.time()
    print(f"Starting Optuna search — {N_TRIALS} trials, 2 hour limit")
    print(f"Optimizing: epochs, lr, momentum, weight_decay, dropout_rate, optimizer, scheduler, init\n")

    study.optimize(
        lambda trial: objective(trial, X, y),
        n_trials=N_TRIALS,
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
    print(f"Best config:")
    for k, v in study.best_params.items():
        print(f"  {k:<20} {v}")

if __name__ == "__main__":
    main()
