import os

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _spec(name, random_state):
    if name == "Naive Bayes (Gaussian)":
        return {
            "estimator": GaussianNB(),
            "param_grid": {
                "model__var_smoothing": [
                    1e-12,
                    1e-10,
                    1e-9,
                    1e-8,
                    1e-7,
                    1e-6,
                    1e-5,
                    1e-3,
                ],
            },
            "scaler": StandardScaler(),
            "key_param": "model__var_smoothing",
            "key_param_display": "var_smoothing",
        }
    if name == "Naive Bayes (Multinomial)":
        return {
            "estimator": MultinomialNB(),
            "param_grid": {
                "model__alpha": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "model__fit_prior": [True, False],
            },
            "scaler": MinMaxScaler(),
            "key_param": "model__alpha",
            "key_param_display": "alpha",
        }
    if name == "Naive Bayes (Complement)":
        return {
            "estimator": ComplementNB(),
            "param_grid": {
                "model__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                "model__norm": [True, False],
                "model__fit_prior": [True, False],
            },
            "scaler": MinMaxScaler(),
            "key_param": "model__alpha",
            "key_param_display": "alpha",
        }
    if name == "Naive Bayes (Bernoulli)":
        return {
            "estimator": BernoulliNB(binarize=0.5),
            "param_grid": {
                "model__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                "model__binarize": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
                "model__fit_prior": [True, False],
            },
            "scaler": MinMaxScaler(),
            "key_param": "model__alpha",
            "key_param_display": "alpha",
        }
    if name == "Decision Tree":
        return {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "param_grid": {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [2, 3, 4, 5, 6, 8, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 5],
                "model__max_features": [None, "sqrt", "log2"],
            },
            "scaler": StandardScaler(),
            "key_param": "model__max_depth",
            "key_param_display": "max_depth",
        }
    if name == "LDA":
        return {
            "estimator": LinearDiscriminantAnalysis(),
            "param_grid": [
                {"model__solver": ["svd"]},
                {
                    "model__solver": ["lsqr"],
                    "model__shrinkage": [None, "auto", 0.0, 0.1, 0.5, 0.9],
                },
                {
                    "model__solver": ["eigen"],
                    "model__shrinkage": [None, "auto", 0.0, 0.1, 0.5, 0.9],
                },
            ],
            "scaler": StandardScaler(),
            "key_param": "model__solver",
            "key_param_display": "solver",
        }
    if name == "SVM (RBF)":
        return {
            "estimator": SVC(random_state=random_state),
            "param_grid": [
                {"model__kernel": ["linear"], "model__C": [0.01, 0.1, 1, 10, 100]},
                {
                    "model__kernel": ["rbf"],
                    "model__C": [0.1, 1, 10, 100, 1000],
                    "model__gamma": [0.001, 0.01, 0.1, 1, "scale"],
                },
                {
                    "model__kernel": ["poly"],
                    "model__C": [0.1, 1, 10],
                    "model__gamma": [0.01, 0.1, "scale"],
                    "model__degree": [2, 3],
                },
            ],
            "scaler": StandardScaler(),
            "key_param": "model__C",
            "key_param_display": "C",
        }
    if name == "KNN (k=5)":
        return {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "model__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "scaler": StandardScaler(),
            "key_param": "model__n_neighbors",
            "key_param_display": "n_neighbors",
        }
    raise ValueError(f"Unknown model family: {name}")


class TuningResult:
    def __init__(
        self,
        name,
        estimator,
        param_grid,
        key_param,
        key_param_display,
        search,
        default_score,
    ):
        self.name = name
        self.estimator = estimator
        self.param_grid = param_grid
        self.key_param = key_param
        self.key_param_display = key_param_display
        self.best_params = {
            k.replace("model__", ""): v for k, v in search.best_params_.items()
        }
        self.best_score = float(search.best_score_)
        self.default_score = float(default_score)
        self.cv_results = pd.DataFrame(search.cv_results_)
        self.n_candidates = len(search.cv_results_["params"])
        self.sensitivity = self._sensitivity()

    def _sensitivity(self):
        col = f"param_{self.key_param}"
        if col not in self.cv_results.columns:
            return None
        sub = self.cv_results[[col, "mean_test_score", "std_test_score"]].dropna(
            subset=["mean_test_score"]
        )
        if len(sub) == 0:
            return None
        grouped = (
            sub.groupby(col, dropna=False)
            .agg(mean=("mean_test_score", "mean"), std=("mean_test_score", "std"))
            .reset_index()
            .rename(columns={col: "value"})
        )
        return grouped

    def __str__(self):
        params_str = ", ".join(f"{k}={v}" for k, v in self.best_params.items())
        return (
            f"{self.name}\n"
            f"  candidates     : {self.n_candidates}\n"
            f"  default CV Acc : {self.default_score:.4f}\n"
            f"  best CV Acc    : {self.best_score:.4f}\n"
            f"  improvement    : {self.best_score - self.default_score:+.4f}\n"
            f"  best params    : {params_str}"
        )


class TuningReport:
    def __init__(self, results, cv_folds, n_samples):
        self.results = results
        self.cv_folds = cv_folds
        self.n_samples = n_samples

    def as_estimators(self, random_state=42):
        out = {}
        for r in self.results:
            params = dict(r.best_params)
            name = r.name
            if name == "SVM (RBF)":
                svc = SVC(random_state=random_state, **params)
                out[name] = CalibratedClassifierCV(svc, ensemble=False)
            elif name == "Decision Tree":
                out[name] = DecisionTreeClassifier(random_state=random_state, **params)
            else:
                out[name] = type(r.estimator)(**params)
        return out

    def dataframe(self):
        rows = []
        for r in self.results:
            rows.append(
                {
                    "Model": r.name,
                    "Default CV Acc": r.default_score,
                    "Best CV Acc": r.best_score,
                    "Δ CV Acc": r.best_score - r.default_score,
                    "Candidates": r.n_candidates,
                    "Best params": "; ".join(
                        f"{k}={v}" for k, v in r.best_params.items()
                    ),
                }
            )
        return pd.DataFrame(rows)

    def __str__(self):
        lines = []
        lines.append("=" * 70)
        lines.append("HYPERPARAMETER TUNING REPORT")
        lines.append("=" * 70)
        lines.append(f"Samples for tuning : {self.n_samples} (train subset only)")
        lines.append(f"CV folds           : {self.cv_folds} (StratifiedKFold, shuffle)")
        lines.append("")
        for r in self.results:
            lines.append("-" * 70)
            lines.append(str(r))
        lines.append("=" * 70)
        lines.append("Summary:")
        header = f"{'Model':<28}{'Def Acc':>10}{'Best Acc':>10}{'Δ':>10}{'Cand':>6}"
        lines.append(header)
        lines.append("-" * len(header))
        for r in self.results:
            lines.append(
                f"{r.name:<28}"
                f"{r.default_score:>10.4f}"
                f"{r.best_score:>10.4f}"
                f"{r.best_score - r.default_score:>+10.4f}"
                f"{r.n_candidates:>6}"
            )
        return "\n".join(lines)


def tune_one(name, X, y, cv_folds=5, random_state=42):
    spec = _spec(name, random_state)
    pipeline = Pipeline([("scaler", spec["scaler"]), ("model", spec["estimator"])])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    default_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )

    search = GridSearchCV(
        pipeline,
        spec["param_grid"],
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)

    return TuningResult(
        name=name,
        estimator=spec["estimator"],
        param_grid=spec["param_grid"],
        key_param=spec["key_param"],
        key_param_display=spec["key_param_display"],
        search=search,
        default_score=float(default_scores.mean()),
    )


def tune_all_models(X, y, cv_folds=5, random_state=42, models=None):
    all_names = [
        "Naive Bayes (Gaussian)",
        "Naive Bayes (Multinomial)",
        "Naive Bayes (Complement)",
        "Naive Bayes (Bernoulli)",
        "Decision Tree",
        "LDA",
        "SVM (RBF)",
        "KNN (k=5)",
    ]
    names = models or all_names
    results = [tune_one(n, X, y, cv_folds, random_state) for n in names]
    return TuningReport(results, cv_folds, len(y))


def compare_default_vs_tuned(default_report, tuned_report):
    lines = []
    lines.append("=" * 100)
    lines.append("DEFAULT vs TUNED — test set comparison")
    lines.append("=" * 100)
    header = (
        f"{'Model':<28}"
        f"{'Def Acc':>10}{'Tuned Acc':>10}{'Δ':>9}"
        f"{'Def F1':>10}{'Tuned F1':>10}{'Δ':>9}"
        f"{'Def AUC':>10}{'Tuned AUC':>10}{'Δ':>9}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for d_model in default_report.models:
        t_model = next((m for m in tuned_report.models if m.name == d_model.name), None)
        if t_model is None:
            continue
        d_auc = (
            d_model.roc_auc_ovr_macro
            if d_model.roc_auc_ovr_macro is not None
            else float("nan")
        )
        t_auc = (
            t_model.roc_auc_ovr_macro
            if t_model.roc_auc_ovr_macro is not None
            else float("nan")
        )
        lines.append(
            f"{d_model.name:<28}"
            f"{d_model.accuracy:>10.4f}"
            f"{t_model.accuracy:>10.4f}"
            f"{t_model.accuracy - d_model.accuracy:>+9.4f}"
            f"{d_model.f1_macro:>10.4f}"
            f"{t_model.f1_macro:>10.4f}"
            f"{t_model.f1_macro - d_model.f1_macro:>+9.4f}"
            f"{d_auc:>10.4f}"
            f"{t_auc:>10.4f}"
            f"{t_auc - d_auc:>+9.4f}"
        )
    lines.append("=" * 100)
    return "\n".join(lines)


def save_tuning_csv(report, path, out_dir="comp"):
    os.makedirs(out_dir, exist_ok=True)
    report.dataframe().to_csv(path, index=False)
    return path
