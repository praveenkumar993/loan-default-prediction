import pandas as pd
import numpy as np
import os
import gc
import joblib
import warnings
warnings.filterwarnings('ignore')

import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

import lightgbm as lgb


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode all categorical columns."""
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
    return df


def get_features_and_target(train: pd.DataFrame):
    """Split into features X and target y. Drop ID columns."""
    drop_cols = ['SK_ID_CURR', 'TARGET']
    X = train.drop(columns=[c for c in drop_cols if c in train.columns])
    y = train['TARGET']
    return X, y


def cross_val_auc(model, X: pd.DataFrame,
                  y: pd.Series, n_splits: int = 5) -> float:
    """Stratified K-Fold cross validation — returns mean AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        print(f"    Fold {fold+1}: AUC = {auc:.4f}")
    return float(np.mean(aucs))


def train_baseline(X: pd.DataFrame, y: pd.Series) -> float:
    """
    Train logistic regression baseline — the scorecard model.
    Banks call this a scorecard. Knowing this term matters.
    """
    print("\n--- Logistic Regression Baseline (Scorecard Model) ---")
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ))
    ])

    auc = cross_val_auc(pipeline, X, y, n_splits=3)
    print(f"  Baseline AUC: {auc:.4f}")
    return auc


def optimize_lgbm(X: pd.DataFrame, y: pd.Series,
                  n_trials: int = 30) -> dict:
    """
    Use Optuna to find best LightGBM hyperparameters.
    More modern and efficient than GridSearchCV.
    """
    print(f"\n--- Optuna Hyperparameter Search ({n_trials} trials) ---")

    def objective(trial):
        params = {
            'objective':        'binary',
            'metric':           'auc',
            'verbosity':        -1,
            'boosting_type':    'gbdt',
            'random_state':     42,
            'n_estimators':     trial.suggest_int('n_estimators', 300, 1000),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves':       trial.suggest_int('num_leaves', 20, 150),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'min_child_samples':trial.suggest_int('min_child_samples', 20, 100),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda':       trial.suggest_float('reg_lambda', 0.0, 1.0),
            'class_weight':     'balanced',
        }
        model = lgb.LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        aucs = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)]
            )
            preds = model.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, preds))
        return float(np.mean(aucs))

    study = optuna.create_study(direction='maximize',
                                sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best AUC:    {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def train_final_lgbm(X: pd.DataFrame, y: pd.Series,
                     best_params: dict) -> lgb.LGBMClassifier:
    """
    Train final LightGBM model on full training data
    using best hyperparameters from Optuna.
    """
    print("\n--- Training Final LightGBM Model ---")

    params = {
        'objective':     'binary',
        'metric':        'auc',
        'verbosity':     -1,
        'boosting_type': 'gbdt',
        'random_state':  42,
        'class_weight':  'balanced',
        **best_params
    }

    model = lgb.LGBMClassifier(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        aucs.append(auc)
        print(f"  Fold {fold+1}: AUC = {auc:.4f}")

    mean_auc = float(np.mean(aucs))
    print(f"\n  Final CV AUC: {mean_auc:.4f}")
    return model, mean_auc


def compute_ks_statistic(model, X: pd.DataFrame,
                         y: pd.Series) -> float:
    """
    KS statistic — the metric credit risk teams actually use in banking.
    Measures separation between default and non-default score distributions.
    """
    from scipy.stats import ks_2samp
    preds = model.predict_proba(X)[:, 1]
    default_scores    = preds[y == 1]
    no_default_scores = preds[y == 0]
    ks_stat, _ = ks_2samp(default_scores, no_default_scores)
    return float(ks_stat)


def run_training_pipeline(data_dir: str,
                          models_dir: str,
                          n_trials: int = 30):
    """
    Master training pipeline:
    1. Build features
    2. Baseline logistic regression
    3. Optuna hyperparameter search
    4. Final LightGBM training
    5. Log everything to MLflow
    6. Save model artifact
    """
    from src.data_loader import load_main_tables
    from src.feature_engineering import build_features

    os.makedirs(models_dir, exist_ok=True)

    # Load and build features
    app_train, app_test = load_main_tables(data_dir)
    train, test = build_features(app_train, app_test, data_dir)
    del app_train, app_test
    gc.collect()

    # Encode categoricals
    train = encode_categoricals(train)
    test  = encode_categoricals(test)

    # Split features and target
    X, y = get_features_and_target(train)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    mlflow.set_experiment("loan-default-prediction")

    with mlflow.start_run(run_name="lgbm_optuna"):

        # Step 1 — baseline
        baseline_auc = train_baseline(X, y)
        mlflow.log_metric("baseline_logreg_auc", baseline_auc)

        # Step 2 — Optuna search
        best_params = optimize_lgbm(X, y, n_trials=n_trials)
        mlflow.log_params(best_params)

        # Step 3 — final model
        model, cv_auc = train_final_lgbm(X, y, best_params)
        mlflow.log_metric("lgbm_cv_auc", cv_auc)

        # Step 4 — KS statistic
        ks = compute_ks_statistic(model, X, y)
        mlflow.log_metric("ks_statistic", ks)
        print(f"\n  KS Statistic: {ks:.4f}")

        # Step 5 — save model
        model_path = os.path.join(models_dir, 'lgbm_model.pkl')
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        print(f"\n  Model saved to {model_path}")

        print(f"\n{'='*50}")
        print(f"  Baseline AUC (LogReg): {baseline_auc:.4f}")
        print(f"  LightGBM CV AUC:       {cv_auc:.4f}")
        print(f"  KS Statistic:          {ks:.4f}")
        print(f"{'='*50}")

    return model, cv_auc, ks


if __name__ == '__main__':
    DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

    model, auc, ks = run_training_pipeline(
        data_dir=DATA_DIR,
        models_dir=MODELS_DIR,
        n_trials=30
    )
    print("\ntrain.py working correctly")