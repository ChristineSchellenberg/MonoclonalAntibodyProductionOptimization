import numpy as np
import pandas as pd

from sklearn.preprocessing import (StandardScaler, PolynomialFeatures)
from sklearn.base import BaseEstimator, TransformerMixin

# =============================================================================
# PHYSICALLY CONSTRAINED POLYNOMIAL FEATURES
# =============================================================================

class TimeConstrainedPolynomial(BaseEstimator, TransformerMixin):

    def __init__(self, degree=3, time_column="Time [h]"):
        self.degree = degree
        self.time_column = time_column

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if self.time_column not in X.columns:
            raise ValueError(
                f"Missing time column: {self.time_column}"
            )

        self.feature_names_ = list(X.columns)

        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            include_bias=True
        )

        self.poly_.fit(X[self.feature_names_])

        return self

    def transform(self, X):
        X_features = X[self.feature_names_]

        time = X_features[self.time_column].to_numpy()

        polynomial_features = self.poly_.transform(X_features)

        return polynomial_features * time[:, np.newaxis]