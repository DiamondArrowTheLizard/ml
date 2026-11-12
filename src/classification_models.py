import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ClassificationModel:
    def __init__(self, name, model, cv_folds):
        self.name = name
        self.model = model
        self.cv_folds = cv_folds

        self.accuracy = None
        self.precision_macro = None
        self.recall_macro = None
        self.f1_macro = None
        self.precision_weighted = None
        self.recall_weighted = None
        self.f1_weighted = None

        self.roc_auc_ovr_macro = None
        self.roc_auc_ovo_macro = None

        self.cv_accuracy_mean = None
        self.cv_accuracy_std = None
        self.cv_f1_mean = None
        self.cv_f1_std = None
        self.cv_auc_mean = None
        self.cv_auc_std = None

        self.y_test = None
        self.y_pred = None
        self.y_proba = None
        self.proba_classes = None
        self.classes = None
        self.confusion = None
        self.report_dict = None

        self.feature_importances = None
        self.coefficients = None

    def evaluate(self, y_true, y_pred, y_proba=None, proba_classes=None):
        self.y_test = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.classes = [int(c) for c in np.unique(np.concatenate([self.y_test, self.y_pred]))]

        self.accuracy = float(accuracy_score(self.y_test, self.y_pred))
        self.precision_macro = float(
            precision_score(self.y_test, self.y_pred, average="macro", zero_division=0)
        )
        self.recall_macro = float(
            recall_score(self.y_test, self.y_pred, average="macro", zero_division=0)
        )
        self.f1_macro = float(
            f1_score(self.y_test, self.y_pred, average="macro", zero_division=0)
        )
        self.precision_weighted = float(
            precision_score(self.y_test, self.y_pred, average="weighted", zero_division=0)
        )
        self.recall_weighted = float(
            recall_score(self.y_test, self.y_pred, average="weighted", zero_division=0)
        )
        self.f1_weighted = float(
            f1_score(self.y_test, self.y_pred, average="weighted", zero_division=0)
        )

        self.confusion = confusion_matrix(self.y_test, self.y_pred, labels=self.classes)
        self.report_dict = classification_report(
            self.y_test, self.y_pred,
            labels=self.classes, zero_division=0, output_dict=True,
        )

        if y_proba is not None and proba_classes is not None:
            self.y_proba = np.asarray(y_proba)
            self.proba_classes = [int(c) for c in proba_classes]
            labels = self.proba_classes
            try:
                self.roc_auc_ovr_macro = float(
                    roc_auc_score(
                        self.y_test, self.y_proba,
                        multi_class="ovr", average="macro", labels=labels,
                    )
                )
            except ValueError:
                self.roc_auc_ovr_macro = None
            try:
                self.roc_auc_ovo_macro = float(
                    roc_auc_score(
                        self.y_test, self.y_proba,
                        multi_class="ovo", average="macro", labels=labels,
                    )
                )
            except ValueError:
                self.roc_auc_ovo_macro = None

    def cross_validate(self, X, y):
        acc = cross_val_score(self.model, X, y, cv=self.cv_folds, scoring="accuracy")
        self.cv_accuracy_mean = float(acc.mean())
        self.cv_accuracy_std = float(acc.std())

        f1 = cross_val_score(self.model, X, y, cv=self.cv_folds, scoring="f1_macro")
        self.cv_f1_mean = float(f1.mean())
        self.cv_f1_std = float(f1.std())

        try:
            auc = cross_val_score(
                self.model, X, y, cv=self.cv_folds, scoring="roc_auc_ovr"
            )
            self.cv_auc_mean = float(auc.mean())
            self.cv_auc_std = float(auc.std())
        except (ValueError, AttributeError):
            self.cv_auc_mean = None
            self.cv_auc_std = None

    def store_importance(self, feature_names):
        est = self.model
        if isinstance(est, Pipeline):
            est = est.named_steps["model"]
        if hasattr(est, "feature_importances_"):
            self.feature_importances = pd.Series(
                est.feature_importances_, index=feature_names
            )
        if hasattr(est, "coef_"):
            coef = np.asarray(est.coef_)
            if coef.ndim == 1:
                coef = coef.reshape(1, -1)
            self.coefficients = pd.DataFrame(coef, columns=feature_names)

    def __str__(self):
        lines = []
        lines.append("-" * 60)
        lines.append(f"Model: {self.name}")
        lines.append("-" * 60)
        lines.append(f"Test Accuracy          : {self.accuracy:.4f}")
        lines.append(f"Test Precision (macro) : {self.precision_macro:.4f}")
        lines.append(f"Test Recall (macro)    : {self.recall_macro:.4f}")
        lines.append(f"Test F1 (macro)        : {self.f1_macro:.4f}")
        lines.append(f"Test F1 (weighted)     : {self.f1_weighted:.4f}")
        if self.roc_auc_ovr_macro is not None:
            lines.append(f"Test AUC-ROC (OvR)     : {self.roc_auc_ovr_macro:.4f}")
        if self.roc_auc_ovo_macro is not None:
            lines.append(f"Test AUC-ROC (OvO)     : {self.roc_auc_ovo_macro:.4f}")
        lines.append(
            f"CV Accuracy            : {self.cv_accuracy_mean:.4f} "
            f"(+/- {self.cv_accuracy_std:.4f})"
        )
        lines.append(
            f"CV F1 (macro)          : {self.cv_f1_mean:.4f} "
            f"(+/- {self.cv_f1_std:.4f})"
        )
        if self.cv_auc_mean is not None:
            lines.append(
                f"CV AUC-ROC (OvR)       : {self.cv_auc_mean:.4f} "
                f"(+/- {self.cv_auc_std:.4f})"
            )
        if self.feature_importances is not None:
            lines.append("Feature importances (top 10):")
            for name, value in (
                self.feature_importances.sort_values(ascending=False).head(10).items()
            ):
                lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


class ClassificationReport:
    def __init__(self, target, feature_names, train_size, test_size, models, label=""):
        self.target = target
        self.feature_names = feature_names
        self.train_size = train_size
        self.test_size = test_size
        self.models = models
        self.label = label
        self.classes = sorted({c for m in models for c in m.classes})

    def best_model(self, metric="accuracy"):
        return max(self.models, key=lambda m: getattr(m, metric))

    def __str__(self):
        lines = []
        lines.append("=" * 60)
        title = "CLASSIFICATION REPORT"
        if self.label:
            title = f"CLASSIFICATION REPORT — {self.label}"
        lines.append(title)
        lines.append("=" * 60)
        lines.append(f"Target variable : {self.target}")
        lines.append(f"Features        : {len(self.feature_names)}")
        lines.append(f"Classes         : {self.classes}")
        lines.append(
            f"Train / Test    : {self.train_size} / {self.test_size} "
            f"({self.train_size / (self.train_size + self.test_size):.0%} / "
            f"{self.test_size / (self.train_size + self.test_size):.0%})"
        )
        lines.append("")
        for model in self.models:
            lines.append(str(model))
            lines.append("")
        best = self.best_model("accuracy")
        lines.append("=" * 60)
        lines.append(
            f"Best model by Accuracy: {best.name} (Accuracy = {best.accuracy:.4f})"
        )
        lines.append("=" * 60)
        return "\n".join(lines)


def prepare_features(dataframe, target):
    y = dataframe[target]
    X = dataframe.drop(columns=[target])
    X = X.select_dtypes(include=["number"])
    return X, y


def _make_pipeline(scaler, estimator):
    return Pipeline(steps=[("scaler", scaler), ("model", estimator)])


def _train_model(
    name, estimator, scaler, X_train, X_test, y_train, y_test, feature_names, cv_folds
):
    pipeline = _make_pipeline(scaler, estimator)
    model = ClassificationModel(name, pipeline, cv_folds)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    y_proba = None
    proba_classes = None
    try:
        y_proba = pipeline.predict_proba(X_test)
        proba_classes = list(pipeline.classes_)
    except (AttributeError, NotImplementedError):
        pass

    model.evaluate(y_test, y_pred, y_proba=y_proba, proba_classes=proba_classes)
    model.cross_validate(X_train, y_train)
    model.store_importance(feature_names)
    return model


def build_classification_models(
    X, y, test_ratio=0.2, cv_folds=5, random_state=42, label=""
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )

    standard_specs = [
        ("Naive Bayes (Gaussian)", GaussianNB()),
        ("Decision Tree", DecisionTreeClassifier(random_state=random_state)),
        ("LDA", LinearDiscriminantAnalysis()),
        ("SVM (RBF)", CalibratedClassifierCV(
            SVC(kernel="rbf", gamma="scale", random_state=random_state),
            ensemble=False,
        )),
        ("KNN (k=5)", KNeighborsClassifier(n_neighbors=5)),
    ]
    minmax_specs = [
        ("Naive Bayes (Multinomial)", MultinomialNB()),
        ("Naive Bayes (Complement)", ComplementNB()),
        ("Naive Bayes (Bernoulli)", BernoulliNB(binarize=0.5)),
    ]

    trained = []
    for name, est in standard_specs:
        trained.append(
            _train_model(
                name, est, StandardScaler(),
                X_train, X_test, y_train, y_test, X.columns, cv_folds,
            )
        )
    for name, est in minmax_specs:
        trained.append(
            _train_model(
                name, est, MinMaxScaler(),
                X_train, X_test, y_train, y_test, X.columns, cv_folds,
            )
        )

    return ClassificationReport(
        target=y.name,
        feature_names=list(X.columns),
        train_size=len(X_train),
        test_size=len(X_test),
        models=trained,
        label=label,
    )