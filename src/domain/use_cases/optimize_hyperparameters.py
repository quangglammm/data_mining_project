from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import xgboost as xgb
from scipy.stats import uniform, randint
import logging

logger = logging.getLogger(__name__)


def tune_hyperparameters(X, y, n_splits=3, n_iter=20):
    """
    Tune XGBoost hyperparameters with TimeSeriesSplit.

    Returns:
        Best parameters dictionary
    """

    param_distributions = {
        "max_depth": randint(3, 7),
        "min_child_weight": randint(2, 8),
        "learning_rate": uniform(0.01, 0.15),
        "subsample": uniform(0.6, 0.3),  # 0.6 to 0.9
        "colsample_bytree": uniform(0.6, 0.3),
        "gamma": uniform(0, 0.3),
        "reg_alpha": uniform(0, 0.5),
        "reg_lambda": uniform(0.5, 2.0),
    }

    base_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    model = xgb.XGBClassifier(**base_params)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=tscv,
        scoring="f1_macro",
        random_state=42,
        verbose=2,
        n_jobs=-1,
    )

    search.fit(X, y)

    logger.info(f"Best F1: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    return search.best_params_
