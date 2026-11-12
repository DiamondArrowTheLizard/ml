import os

import pandas as pd

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1_macro": "F1 (macro)",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
}

NOT_APPLICABLE_ON_PCA = {
    "Naive Bayes (Multinomial)",
    "Naive Bayes (Complement)",
}


class ModelComparison:
    def __init__(self, report_a, report_b):
        self.report_a = report_a
        self.report_b = report_b
        self.rows = self._build_rows()
        self.not_applicable = self._not_applicable()

    def _build_rows(self):
        rows = []
        for model_a in self.report_a.models:
            if model_a.name in NOT_APPLICABLE_ON_PCA:
                continue
            model_b = next(
                (m for m in self.report_b.models if m.name == model_a.name), None
            )
            if model_b is None:
                continue
            rows.append(
                {
                    "Model": model_a.name,
                    "Accuracy (raw)": model_a.accuracy,
                    "Accuracy (PCA)": model_b.accuracy,
                    "ΔAccuracy": model_b.accuracy - model_a.accuracy,
                    "F1 macro (raw)": model_a.f1_macro,
                    "F1 macro (PCA)": model_b.f1_macro,
                    "ΔF1 macro": model_b.f1_macro - model_a.f1_macro,
                    "CV Accuracy (raw)": model_a.cv_accuracy_mean,
                    "CV Accuracy (PCA)": model_b.cv_accuracy_mean,
                    "ΔCV Accuracy": model_b.cv_accuracy_mean - model_a.cv_accuracy_mean,
                }
            )
        return rows

    def _not_applicable(self):
        return sorted(
            m.name for m in self.report_a.models
            if m.name in NOT_APPLICABLE_ON_PCA
        )

    def dataframe(self):
        return pd.DataFrame(self.rows)

    def best(self, label, metric="accuracy"):
        report = self.report_a if label == "raw" else self.report_b
        candidates = [m for m in report.models if m.name not in NOT_APPLICABLE_ON_PCA]
        return max(candidates, key=lambda m: getattr(m, metric))

    def __str__(self):
        lines = []
        lines.append("=" * 100)
        lines.append("MODEL COMPARISON — RAW FEATURES vs PCA COMPONENTS")
        lines.append("=" * 100)
        lines.append(
            f"Raw features : {len(self.report_a.feature_names)} "
            f"({self.report_a.label or 'raw'})"
        )
        lines.append(
            f"PCA features : {len(self.report_b.feature_names)} "
            f"({self.report_b.label or 'pca'})"
        )
        lines.append("")

        header = (
            f"{'Model':<28}"
            f"{'Acc raw':>10}{'Acc pca':>10}{'ΔAcc':>10}"
            f"{'F1 raw':>10}{'F1 pca':>10}{'ΔF1':>10}"
            f"{'CVAcc raw':>12}{'CVAcc pca':>12}{'ΔCVAcc':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in self.rows:
            lines.append(
                f"{row['Model']:<28}"
                f"{row['Accuracy (raw)']:>10.4f}"
                f"{row['Accuracy (PCA)']:>10.4f}"
                f"{row['ΔAccuracy']:>+10.4f}"
                f"{row['F1 macro (raw)']:>10.4f}"
                f"{row['F1 macro (PCA)']:>10.4f}"
                f"{row['ΔF1 macro']:>+10.4f}"
                f"{row['CV Accuracy (raw)']:>12.4f}"
                f"{row['CV Accuracy (PCA)']:>12.4f}"
                f"{row['ΔCV Accuracy']:>+10.4f}"
            )

        if self.not_applicable:
            lines.append("")
            lines.append("Not applicable on PCA (count-based NB):")
            for name in self.not_applicable:
                lines.append(f"  - {name}")

        lines.append("")
        lines.append("Winners per metric (test set):")
        for metric, label in METRIC_LABELS.items():
            best_raw = max((m for m in self.report_a.models if m.name not in NOT_APPLICABLE_ON_PCA), key=lambda m: getattr(m, metric))
            best_pca = max((m for m in self.report_b.models if m.name not in NOT_APPLICABLE_ON_PCA), key=lambda m: getattr(m, metric))
            overall = (
                "RAW"
                if getattr(best_raw, metric) >= getattr(best_pca, metric)
                else "PCA"
            )
            lines.append(
                f"  {label:<20}  RAW: {best_raw.name} = "
                f"{getattr(best_raw, metric):.4f} | "
                f"PCA: {best_pca.name} = {getattr(best_pca, metric):.4f}  "
                f"→ winner: {overall}"
            )

        lines.append("")
        best_raw = self.best("raw", "accuracy")
        best_pca = self.best("pca", "accuracy")
        lines.append(
            f"Best overall on test accuracy — RAW: {best_raw.name} "
            f"({best_raw.accuracy:.4f})  vs  PCA: {best_pca.name} "
            f"({best_pca.accuracy:.4f})"
        )
        lines.append("=" * 100)
        return "\n".join(lines)


def save_comparison_csv(comparison, path, out_dir="comp"):
    os.makedirs(out_dir, exist_ok=True)
    comparison.dataframe().to_csv(path, index=False)
    return path