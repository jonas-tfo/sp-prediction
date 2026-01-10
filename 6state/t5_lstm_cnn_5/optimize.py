
import optuna

from .model import SPCNNClassifier
from .config import Config
from .train import train


def objective(trial):

    LR = trial.suggest_float("lr", 1e-5, 1e-1, log=True),
    WEIGHT_DECAY = trial.suggest_float("weight_decay", 0.0, 0.1),
    DROPOUT = trial.suggest_float("dropout", 0.1, 0.5),
    # LSTM_HIDDEN = trial.suggest_categorical("lstm_hidden", [256, 512]),
    # LSTM_LAYERS = trial.suggest_categorical("lstm_layers", [1, 2]),
    # CONV_FILTERS = trial.suggest_categorical("conv_filters", [256, 512]),

    # Run with these hyperparameters
    avg_mcc = train(lr=LR, weight_decay=WEIGHT_DECAY, dropout=DROPOUT)

    return avg_mcc


def study():
    # want to maximize mcc
    study = optuna.create_study(direction="maximize")

    # todo tweak trial num
    study.optimize(objective, n_trials=30)

    # Print results
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest MCC: {study.best_value:.4f}")


if __name__ == "__main__":
    study()

