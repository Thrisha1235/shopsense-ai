"""Load, validate, clean and prepare the UCI Online Shoppers Purchasing Intention data.

Leakage rules this module follows
---------------------------------
* The original CSV is only ever *read*. Its SHA-256 is checked against the checksum
  recorded at download time, so accidental edits are detected.
* ``clean()`` applies only **stateless, row-wise** fixes (renaming a month label,
  casting types, dropping rows with no target). Nothing in it learns a statistic
  from the data, so it is safe to run before splitting.
* Anything that *learns* from data (median imputation, scaling, one-hot categories)
  lives in the scikit-learn ``ColumnTransformer`` returned by ``build_preprocessor()``.
  It is returned **unfitted** and must be fitted on the training split only, ideally
  inside a ``Pipeline`` together with the model.
* The target ``Revenue`` is never used as a feature.

Run from the backend directory to validate the data and preview the split:
    python -m ml.data_preprocessing
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_CSV_PATH = DATA_DIR / "online_shoppers_intention.csv"
CHECKSUM_PATH = DATA_DIR / "online_shoppers_intention.sha256"

# -------------------------------------------------------------------------- schema
TARGET = "Revenue"

# Page-visit counts and time spent (seconds); heavily right-skewed.
COUNT_DURATION_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
]
# Google Analytics metrics for the session.
RATE_FEATURES = ["BounceRates", "ExitRates"]
VALUE_FEATURES = ["PageValues"]
# Closeness of the visit to a special day (0, 0.2, ... 1.0).
SPECIAL_DAY_FEATURES = ["SpecialDay"]

NUMERIC_FEATURES = (
    COUNT_DURATION_FEATURES + RATE_FEATURES + VALUE_FEATURES + SPECIAL_DAY_FEATURES
)
# Integer *codes* with no numeric meaning (e.g. Browser 3 is not "more" than Browser 2).
CODED_CATEGORICAL_FEATURES = ["OperatingSystems", "Browser", "Region", "TrafficType"]
TEXT_CATEGORICAL_FEATURES = ["Month", "VisitorType"]
CATEGORICAL_FEATURES = TEXT_CATEGORICAL_FEATURES + CODED_CATEGORICAL_FEATURES
BOOLEAN_FEATURES = ["Weekend"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES
EXPECTED_COLUMNS = FEATURES + [TARGET]

# Skewed features that get a log1p transform before scaling.
LOG_FEATURES = COUNT_DURATION_FEATURES + VALUE_FEATURES

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# The raw file spells June in full while all other months are abbreviated.
MONTH_ALIASES = {"June": "Jun"}
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]

_TRUE_STRINGS = {"true", "1", "yes"}
_FALSE_STRINGS = {"false", "0", "no"}


class DataValidationError(ValueError):
    """Raised when the dataset violates the expected schema or value ranges."""


# ------------------------------------------------------------------------- loading
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path = RAW_CSV_PATH, checksum_path: Path = CHECKSUM_PATH) -> bool | None:
    """Return True/False if a recorded checksum exists, None if none was recorded."""
    if not Path(checksum_path).exists():
        return None
    expected = Path(checksum_path).read_text(encoding="utf-8").split()[0].strip().lower()
    return sha256_of(path) == expected


def load_raw(path: Path = RAW_CSV_PATH, *, verify: bool = True) -> pd.DataFrame:
    """Read the original CSV (read-only) and optionally verify its checksum."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python -I ml/download_data.py` first."
        )
    if verify and verify_checksum(path) is False:
        raise DataValidationError(
            f"{path.name} does not match its recorded SHA-256 - the original file was modified."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------- validation
@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    missing_values: dict[str, int]
    n_duplicate_rows: int
    target_counts: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise DataValidationError("Dataset validation failed:\n- " + "\n- ".join(self.errors))

    def summary(self) -> str:
        lines = [
            f"rows={self.n_rows:,}  columns={self.n_columns}",
            f"missing values total={sum(self.missing_values.values())}",
            f"duplicate rows={self.n_duplicate_rows}",
            f"target counts={self.target_counts}",
        ]
        lines += [f"ERROR: {e}" for e in self.errors]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce True/False, 1/0 or 'TRUE'/'FALSE' strings to a nullable boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    lowered = series.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out[lowered.isin(_TRUE_STRINGS)] = True
    out[lowered.isin(_FALSE_STRINGS)] = False
    return out


def validate(df: pd.DataFrame) -> ValidationReport:
    """Check schema, types and value ranges. Does not modify ``df``."""
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
    if extra_cols:
        warnings.append(f"unexpected extra columns (ignored): {extra_cols}")

    present = [c for c in EXPECTED_COLUMNS if c in df.columns]
    missing_values = {c: int(df[c].isna().sum()) for c in present}

    for c in [c for c in NUMERIC_FEATURES + CODED_CATEGORICAL_FEATURES if c in df.columns]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            errors.append(f"{c} should be numeric but has dtype {df[c].dtype}")
            continue
        if (df[c] < 0).any():
            errors.append(f"{c} has {int((df[c] < 0).sum())} negative values")

    for c in [c for c in RATE_FEATURES + SPECIAL_DAY_FEATURES if c in df.columns]:
        if pd.api.types.is_numeric_dtype(df[c]) and (df[c] > 1).any():
            errors.append(f"{c} has {int((df[c] > 1).sum())} values above 1")

    if "Month" in df.columns:
        months = df["Month"].dropna().astype(str).replace(MONTH_ALIASES)
        unknown = sorted(set(months) - set(MONTH_ORDER))
        if unknown:
            errors.append(f"Month has unknown values: {unknown}")

    if "VisitorType" in df.columns:
        unknown = sorted(set(df["VisitorType"].dropna().astype(str)) - set(VISITOR_TYPES))
        if unknown:
            errors.append(f"VisitorType has unknown values: {unknown}")

    target_counts: dict[str, int] = {}
    for c in [c for c in BOOLEAN_FEATURES + [TARGET] if c in df.columns]:
        coerced = _as_bool(df[c])
        bad = int((coerced.isna() & df[c].notna()).sum())
        if bad:
            errors.append(f"{c} has {bad} values that are not boolean")
        if c == TARGET:
            target_counts = {str(k): int(v) for k, v in coerced.value_counts().items()}
            if missing_values.get(TARGET):
                warnings.append(f"{missing_values[TARGET]} rows have no target and will be dropped")
            if len(target_counts) < 2:
                errors.append("target has fewer than two classes")

    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        warnings.append(f"{n_dupes} fully duplicated rows (kept by default, see clean())")
    feature_missing = sum(v for k, v in missing_values.items() if k != TARGET)
    if feature_missing:
        warnings.append(f"{feature_missing} missing feature values (imputed inside the pipeline)")

    return ValidationReport(
        n_rows=len(df),
        n_columns=df.shape[1],
        missing_values=missing_values,
        n_duplicate_rows=n_dupes,
        target_counts=target_counts,
        errors=errors,
        warnings=warnings,
    )


# ------------------------------------------------------------------------ cleaning
def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature normalisation shared by training (clean) and inference.

    Works on any subset of feature columns and never needs the target. Returns a copy.
    """
    out = df.copy()
    if "Month" in out:
        month = out["Month"].astype("string").str.strip().str.title()
        out["Month"] = month.replace(MONTH_ALIASES)
    if "VisitorType" in out:
        out["VisitorType"] = out["VisitorType"].astype("string").str.strip()
    for c in CODED_CATEGORICAL_FEATURES:
        if c in out:
            out[c] = out[c].astype("Int64")
    if "Weekend" in out:
        out["Weekend"] = _as_bool(out["Weekend"])
    return out


def clean(df: pd.DataFrame, *, drop_duplicates: bool = False) -> pd.DataFrame:
    """Stateless, row-wise cleaning. Returns a new DataFrame; the input is untouched.

    * normalises the Month label ``June`` -> ``Jun``
    * casts ``Weekend`` to bool and ``Revenue`` to int (0/1)
    * casts coded categoricals to pandas nullable integers
    * drops rows whose target is missing (they cannot be used for supervision)
    * optionally drops exact duplicate rows. Off by default: in this dataset every
      duplicate is a zero-duration, single-page bounce session, which is a plausible
      real, repeated behaviour rather than a data-entry error.

    Missing *feature* values are deliberately left as NaN so the imputer inside the
    pipeline can learn its fill values from the training split only.
    """
    out = clean_features(df.loc[:, [c for c in EXPECTED_COLUMNS if c in df.columns]])

    target = _as_bool(out[TARGET])
    out = out.loc[target.notna()].copy()
    out[TARGET] = target.loc[out.index].astype(int)

    if drop_duplicates:
        out = out.drop_duplicates()

    return out.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURES].copy(), df[TARGET].copy()


def make_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split so both sets keep the ~15% purchase rate."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ------------------------------------------------------------------- preprocessing
def _to_float(X):
    """Convert pandas nullable dtypes (pd.NA) to float64 with NaN for sklearn."""
    return np.asarray(pd.DataFrame(X).astype("float64"))


def _to_object(X):
    """Convert to plain object values with NaN for missing, for the categorical branch."""
    frame = pd.DataFrame(X).astype(object)
    return frame.where(frame.notna(), np.nan).to_numpy()


def build_preprocessor(
    *,
    scale_numeric: bool = True,
    log_transform_skewed: bool = True,
    exclude: tuple[str, ...] = (),
) -> ColumnTransformer:
    """Return an UNFITTED ColumnTransformer. Fit it on training data only.

    Numeric  : median impute -> log1p (skewed counts/durations/PageValues) -> scale
    Category : most-frequent impute -> one-hot (unseen categories ignored)
    Boolean  : most-frequent impute -> 0/1

    ``exclude`` drops features, e.g. ``exclude=("PageValues",)`` for a leakage
    sensitivity check (see reports/EDA_SUMMARY.md).
    """
    numeric = [c for c in NUMERIC_FEATURES if c not in exclude]
    log_cols = [c for c in numeric if c in LOG_FEATURES] if log_transform_skewed else []
    linear_cols = [c for c in numeric if c not in log_cols]
    categorical = [c for c in CATEGORICAL_FEATURES if c not in exclude]
    boolean = [c for c in BOOLEAN_FEATURES if c not in exclude]

    def numeric_pipe(log: bool) -> Pipeline:
        steps = [
            ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median")),
        ]
        if log:
            steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    categorical_pipe = Pipeline([
        ("to_object", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    boolean_pipe = Pipeline([
        ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    transformers = []
    if log_cols:
        transformers.append(("num_log", numeric_pipe(log=True), log_cols))
    if linear_cols:
        transformers.append(("num", numeric_pipe(log=False), linear_cols))
    if categorical:
        transformers.append(("cat", categorical_pipe, categorical))
    if boolean:
        transformers.append(("bool", boolean_pipe, boolean))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


# ------------------------------------------------------------------- entry points
@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer  # UNFITTED
    report: ValidationReport


def prepare_datasets(
    path: Path = RAW_CSV_PATH,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_duplicates: bool = False,
    exclude: tuple[str, ...] = (),
) -> PreparedData:
    """Load -> validate -> clean -> split, and return an unfitted preprocessor."""
    raw = load_raw(path)
    report = validate(raw)
    report.raise_if_invalid()
    df = clean(raw, drop_duplicates=drop_duplicates)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = make_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return PreparedData(
        X_train, X_test, y_train, y_test,
        build_preprocessor(exclude=exclude),
        report,
    )


def main() -> None:
    data = prepare_datasets()
    print("== Validation ==")
    print(data.report.summary())
    print(f"checksum verified: {verify_checksum()}")

    print("\n== Train/test split (stratified) ==")
    for name, y in [("train", data.y_train), ("test", data.y_test)]:
        print(f"{name:5s} rows={len(y):,}  purchase rate={y.mean():.4f}")

    # Demonstration only: fit on TRAIN, then transform test with train statistics.
    pre = data.preprocessor
    Xt_train = pre.fit_transform(data.X_train)
    Xt_test = pre.transform(data.X_test)
    print("\n== Preprocessor (fitted on train only) ==")
    print(f"train matrix {Xt_train.shape}, test matrix {Xt_test.shape}")
    print(f"NaNs after transform: train={int(np.isnan(Xt_train).sum())} test={int(np.isnan(Xt_test).sum())}")
    print(f"output features: {len(pre.get_feature_names_out())}")


if __name__ == "__main__":
    main()
