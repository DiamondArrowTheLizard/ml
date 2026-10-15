import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RegressionModel:
    def __init__(self, name: str, model, cv_folds: int):
        self.name = name
        self.model = model
        self.cv_folds = cv_folds

        self.rmse: float = None
        self.r2: float = None
        self.mape: float = None

        self.cv_rmse_mean: float = None
        self.cv_rmse_std: float = None
        self.cv_r2_mean: float = None
        self.cv_r2_std: float = None

        self.coefficients: pd.Series = None
        self.intercept: float = None

    def evaluate(self, y_true, y_pred):
        self.rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        self.r2 = float(r2_score(y_true, y_pred))
        self.mape = float(mean_absolute_percentage_error(y_true, y_pred))

    def cross_validate(self, X, y):
        neg_rmse = cross_val_score(
            self.model,
            X,
            y,
            cv=self.cv_folds,
            scoring="neg_root_mean_squared_error",
        )
        self.cv_rmse_mean = float(-neg_rmse.mean())
        self.cv_rmse_std = float(neg_rmse.std())

        r2 = cross_val_score(
            self.model,
            X,
            y,
            cv=self.cv_folds,
            scoring="r2",
        )
        self.cv_r2_mean = float(r2.mean())
        self.cv_r2_std = float(r2.std())

    def store_coefficients(self, feature_names):
        estimator = self.model
        if isinstance(estimator, TransformedTargetRegressor):
            estimator = estimator.regressor_
        if isinstance(estimator, Pipeline):
            estimator = estimator.named_steps["model"]
        if hasattr(estimator, "coef_"):
            self.coefficients = pd.Series(estimator.coef_, index=feature_names)
        if hasattr(estimator, "intercept_"):
            self.intercept = float(estimator.intercept_)

    def __str__(self):
        lines = []
        lines.append("-" * 60)
        lines.append(f"Model: {self.name}")
        lines.append("-" * 60)
        lines.append(f"Test RMSE : {self.rmse:,.2f}")
        lines.append(f"Test R²   : {self.r2:.4f}")
        lines.append(f"Test MAPE : {self.mape:.4f} ({self.mape * 100:.2f}%)")
        lines.append(
            f"CV RMSE   : {self.cv_rmse_mean:,.2f} (+/- {self.cv_rmse_std:,.2f})"
        )
        lines.append(f"CV R²     : {self.cv_r2_mean:.4f} (+/- {self.cv_r2_std:.4f})")
        if self.intercept is not None:
            lines.append(f"Intercept : {self.intercept:,.2f}")
        if self.coefficients is not None:
            lines.append("Coefficients:")
            for name, value in self.coefficients.items():
                lines.append(f"  {name}: {value:,.2f}")
        return "\n".join(lines)


class RegressionReport:
    def __init__(
        self,
        target: str,
        feature_names: list,
        train_size: int,
        test_size: int,
        models: list,
    ):
        self.target = target
        self.feature_names = feature_names
        self.train_size = train_size
        self.test_size = test_size
        self.models = models

    def best_model(self, metric: str = "rmse"):
        reverse = metric in ("r2",)
        return sorted(
            self.models,
            key=lambda m: getattr(m, metric),
            reverse=reverse,
        )[0]

    def __str__(self):
        lines = []
        lines.append("=" * 60)
        lines.append("REGRESSION REPORT")
        lines.append("=" * 60)
        lines.append(f"Target variable : {self.target}")
        lines.append(f"Features        : {', '.join(self.feature_names)}")
        lines.append(
            f"Train / Test    : {self.train_size} / {self.test_size} "
            f"({self.train_size / (self.train_size + self.test_size):.0%} / "
            f"{self.test_size / (self.train_size + self.test_size):.0%})"
        )
        lines.append("")
        for model in self.models:
            lines.append(str(model))
            lines.append("")
        best = self.best_model("rmse")
        lines.append("=" * 60)
        lines.append(f"Best model by RMSE: {best.name} (RMSE = {best.rmse:,.2f})")
        lines.append("=" * 60)
        return "\n".join(lines)


def prepare_features(dataframe: pd.DataFrame, target: str):
    df = dataframe.copy()
    y = df[target].astype(float)
    X = df.drop(columns=[target])
    X = pd.get_dummies(X, drop_first=True)
    X = X.astype(float)
    return X, y


def _wrap(estimator):
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        transformer=StandardScaler(),
    )


def build_regression_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> RegressionReport:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )

    specs = [
        ("Linear Regression", LinearRegression()),
        (
            "Lasso Regression",
            Lasso(alpha=0.1, max_iter=100000, tol=1e-3, random_state=random_state),
        ),
        ("Ridge Regression", Ridge(alpha=1.0, random_state=random_state)),
    ]

    trained_models = []
    for name, estimator in specs:
        wrapped = _wrap(estimator)
        model = RegressionModel(name, wrapped, cv_folds)

        wrapped.fit(X_train, y_train)
        y_pred = wrapped.predict(X_test)

        model.evaluate(y_test, y_pred)
        model.cross_validate(X_train, y_train)
        model.store_coefficients(X.columns)

        trained_models.append(model)

    return RegressionReport(
        target=y.name,
        feature_names=list(X.columns),
        train_size=len(X_train),
        test_size=len(X_test),
        models=trained_models,
    )
