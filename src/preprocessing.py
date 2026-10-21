"""
Preprocessing Module for DemandWise
===================================
Constructs Scikit-learn ColumnTransformer and Pipeline.
Handles missing values with SimpleImputer, scales numeric features,
and one-hot encodes categorical features with robust unknown handling.
"""

from typing import List, Tuple, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from src.feature_engineering import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_MODEL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Creates a ColumnTransformer that imputes and scales numeric features,
    and one-hot encodes categorical variables.
    Unknown categories in inference or test sets are handled safely (all zeros).
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERIC_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )


def build_model_pipeline(
    class_weight: Optional[str] = 'balanced',
    C: float = 1.0,
    random_state: int = 42
) -> Pipeline:
    """
    Builds the full Scikit-learn Pipeline with Preprocessor and LogisticRegression classifier.
    """
    preprocessor = build_preprocessor()
    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver='lbfgs',
        max_iter=1000,
        random_state=random_state
    )
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
