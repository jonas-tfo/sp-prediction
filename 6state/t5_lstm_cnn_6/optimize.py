
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .config import Config
from .train import train


def objective(trial):
    """Objective function for hyperparameter optimization."""

    # Core training hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    # Architecture hyperparameters
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [256, 512])
    lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2, 3])
    conv_filters = trial.suggest_categorical("conv_filters", [256, 512])

    # Update config with sampled values
    Config.LR = lr
    Config.WEIGHT_DECAY = weight_decay
    Config.DROPOUT = dropout
    Config.LSTM_HIDDEN = lstm_hidden
    Config.LSTM_LAYERS = lstm_layers
    Config.CONV_FILTERS = conv_filters

    # Run training with these hyperparameters
    avg_mcc = train(lr=lr, weight_decay=weight_decay, dropout=dropout)

    return avg_mcc


def run_optimization(n_trials=50, study_name="t5_lstm_cnn_6_optimization"):
    """Run hyperparameter optimization with Optuna."""

    # Create study with TPE sampler and median pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///{Config.OUTPUT_DIR}/optuna_study.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)

    print(f"\nNumber of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (MCC): {best_trial.value:.4f}")

    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Save best params to file
    best_params_path = Config.OUTPUT_DIR / "best_hyperparams.txt"
    with open(best_params_path, "w") as f:
        f.write(f"Best MCC: {best_trial.value:.4f}\n\n")
        f.write("Best hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    print(f"\nBest parameters saved to: {best_params_path}")

    return study


if __name__ == "__main__":
    run_optimization(n_trials=20)
