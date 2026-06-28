import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config_entity import DataTransformationConfig
from churn.logger import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    @staticmethod
    def _build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
            ]
        )

    def run(self, data_path: str):
        logger.info("===== Stage: Data Transformation =====")
        df = pd.read_csv(data_path)

        # 1. clean: coerce TotalCharges to numeric (the 11 blanks -> NaN, imputed below)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            n_nan = int(df["TotalCharges"].isna().sum())
            logger.info(f"Coerced TotalCharges to numeric; {n_nan} NaN(s) will be imputed")

        # 2. drop identifier columns that carry no predictive signal
        drop = [c for c in self.config.drop_columns if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
            logger.info(f"Dropped columns: {drop}")

        # 3. separate features / target, encode target (Yes -> 1, No -> 0)
        target = self.config.target_column
        y = (df[target] == "Yes").astype(int)
        X = df.drop(columns=[target])

        # 4. column groups by dtype (numeric scaled, categorical one-hot encoded)
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
        categorical_cols = X.select_dtypes(include="object").columns.tolist()
        logger.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
        logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

        # 5. split BEFORE fitting anything (prevents leakage), stratified on target
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        logger.info(
            f"Split -> train: {X_train.shape[0]} rows, test: {X_test.shape[0]} rows | "
            f"train churn rate: {y_train.mean():.3f}, test churn rate: {y_test.mean():.3f}"
        )

        # 6. fit the preprocessor on TRAIN only, then transform both splits
        preprocessor = self._build_preprocessor(numeric_cols, categorical_cols)
        X_train_t = preprocessor.fit_transform(X_train)
        X_test_t = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out()
        logger.info(f"Features after encoding: {len(feature_names)}")

        # 7. persist the fitted preprocessor + processed train/test sets
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, self.config.preprocessor_path)

        train_df = pd.DataFrame(X_train_t, columns=feature_names)
        train_df[target] = y_train.to_numpy()
        test_df = pd.DataFrame(X_test_t, columns=feature_names)
        test_df[target] = y_test.to_numpy()
        train_df.to_csv(self.config.train_path, index=False)
        test_df.to_csv(self.config.test_path, index=False)

        logger.info(f"Saved preprocessor -> {self.config.preprocessor_path}")
        logger.info(f"Saved train set   -> {self.config.train_path} {train_df.shape}")
        logger.info(f"Saved test set    -> {self.config.test_path} {test_df.shape}")
        return str(self.config.train_path), str(self.config.test_path)
