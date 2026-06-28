import json

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from xgboost import XGBClassifier

from churn.config_entity import ModelTrainerConfig
from churn.logger import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def _candidate_models(self, scale_pos_weight: float) -> dict:
        rs = self.config.random_state
        # imbalance handled directly: class_weight for sklearn models,
        # scale_pos_weight (= negatives / positives) for XGBoost
        return {
            "logistic_regression": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=rs),
            "random_forest": RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=rs, n_jobs=-1),
            "xgboost": XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                random_state=rs, n_jobs=-1),
        }

    @staticmethod
    def _evaluate(y_true, y_pred, y_proba) -> dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
        }

    def run(self):
        logger.info("===== Stage: Model Training =====")
        target = self.config.target_column
        train_df = pd.read_csv(self.config.train_path)
        test_df = pd.read_csv(self.config.test_path)

        X_train, y_train = train_df.drop(columns=[target]), train_df[target]
        X_test, y_test = test_df.drop(columns=[target]), test_df[target]

        neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
        scale_pos_weight = neg / pos if pos else 1.0
        logger.info(f"Train class balance -> negatives: {neg}, positives: {pos}")

        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.config.experiment_name)
        results = {}
        best_name, best_model, best_auc = None, None, -1.0

        for name, model in self._candidate_models(scale_pos_weight).items():
            with mlflow.start_run(run_name=name):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics = self._evaluate(y_test, y_pred, y_proba)

                mlflow.log_param("model_type", name)
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                # native flavor per model (skops, MLflow's default sklearn
                # serializer, won't trust XGBoost's types)
                if name == "xgboost":
                    mlflow.xgboost.log_model(model, name="model")
                else:
                    mlflow.sklearn.log_model(model, name="model")

                results[name] = metrics
                logger.info(f"{name:>20} | roc_auc={metrics['roc_auc']:.4f} "
                            f"f1={metrics['f1']:.4f} recall={metrics['recall']:.4f}")

                if metrics["roc_auc"] > best_auc:
                    best_name, best_model, best_auc = name, model, metrics["roc_auc"]

        logger.info(f"Best model: {best_name} (roc_auc={best_auc:.4f})")
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, self.config.model_path)
        with open(self.config.metrics_path, "w") as f:
            json.dump({"best_model": best_name, "metrics_by_model": results}, f, indent=4)
        logger.info(f"Saved best model  -> {self.config.model_path}")
        logger.info(f"Saved metrics     -> {self.config.metrics_path}")
        return best_name, results