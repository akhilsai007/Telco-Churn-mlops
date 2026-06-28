import pandas as pd

from churn.config_entity import DataValidationConfig
from churn.logger import logger
from churn.utils import save_json

# dtype names vary across pandas versions (e.g. object vs str in pandas 3.x),
# so compare by equivalence group rather than exact string.
_DTYPE_GROUPS = {
    "object": "text", "str": "text", "string": "text",
    "int64": "int", "int32": "int", "Int64": "int", "Int32": "int",
    "float64": "float", "float32": "float", "Float64": "float", "Float32": "float",
    "bool": "bool", "boolean": "bool",
}


def _norm_dtype(dtype) -> str:
    return _DTYPE_GROUPS.get(str(dtype), str(dtype))


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def run(self, data_path: str) -> bool:
        logger.info("===== Stage: Data Validation =====")
        df = pd.read_csv(data_path)

        report: dict = {}
        errors: list = []
        warnings: list = []

        # 1. column presence (hard checks)
        expected = set(self.config.all_schema.keys())
        actual = set(df.columns)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        report["missing_columns"] = missing
        report["unexpected_columns"] = unexpected
        if missing:
            errors.append(f"Missing columns: {missing}")
        if unexpected:
            errors.append(f"Unexpected columns: {unexpected}")

        # 2. dtype checks (soft - inference can vary, so recorded as warnings)
        dtype_mismatches = {}
        for col, expected_dtype in self.config.all_schema.items():
            if col in df.columns and _norm_dtype(df[col].dtype) != _norm_dtype(expected_dtype):
                dtype_mismatches[col] = {
                    "expected": expected_dtype,
                    "actual": str(df[col].dtype),
                }
        report["dtype_mismatches"] = dtype_mismatches
        if dtype_mismatches:
            warnings.append(f"Dtype mismatches: {list(dtype_mismatches)}")

        # 3. target column present with expected classes (hard check)
        target = self.config.target_column
        if target not in df.columns:
            errors.append(f"Target column '{target}' is missing")
        else:
            classes = sorted(df[target].dropna().unique().tolist())
            report["target_classes"] = classes
            allowed = self.config.categorical_values.get(target)
            if allowed and not set(classes).issubset(set(allowed)):
                errors.append(
                    f"Target '{target}' has unexpected classes {classes}, "
                    f"allowed {allowed}"
                )

        # 4. categorical values within the allowed set (hard check)
        invalid = {}
        for col, allowed in self.config.categorical_values.items():
            if col in df.columns:
                unknown = sorted(set(df[col].dropna().unique()) - set(allowed))
                if unknown:
                    invalid[col] = unknown
        report["invalid_categorical_values"] = invalid
        if invalid:
            errors.append(f"Invalid categorical values in: {list(invalid)}")

        # 5. missing-value report (informational)
        report["missing_value_counts"] = {
            c: int(n) for c, n in df.isna().sum().items() if n > 0
        }

        # 6. known data-quality issue: TotalCharges blanks / non-numeric (warning)
        if "TotalCharges" in df.columns:
            coerced = pd.to_numeric(df["TotalCharges"], errors="coerce")
            non_numeric = int((coerced.isna() & df["TotalCharges"].notna()).sum())
            report["totalcharges_non_numeric"] = non_numeric
            if non_numeric:
                warnings.append(
                    f"TotalCharges has {non_numeric} non-numeric/blank value(s) "
                    f"- handle these in preprocessing"
                )

        # 7. row / column counts
        report["n_rows"] = int(df.shape[0])
        report["n_columns"] = int(df.shape[1])
        if df.shape[0] == 0:
            errors.append("Dataset is empty (0 rows)")
        if self.config.expected_rows and df.shape[0] != self.config.expected_rows:
            warnings.append(
                f"Row count {df.shape[0]} != expected {self.config.expected_rows}"
            )

        status = len(errors) == 0
        report["errors"] = errors
        report["warnings"] = warnings
        report["validation_status"] = "PASS" if status else "FAIL"

        save_json(self.config.report_path, report)
        self.config.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.status_path.write_text(report["validation_status"])

        for w in warnings:
            logger.warning(w)
        if not status:
            for e in errors:
                logger.error(e)
            raise ValueError(f"Data validation FAILED: {errors}")

        logger.info("Data validation PASSED")
        return status
