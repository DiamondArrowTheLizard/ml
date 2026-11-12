import os

import pandas as pd

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "roc_auc_ovr_macro": "AUC-ROC (OvR, macro)",
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
            auc_raw = model_a.roc_auc_ovr_macro
            auc_pca = model_b.roc_auc_ovr_macro
            if auc_raw is not None and auc_pca is not None:
                auc_raw_v = float(auc_raw)
                auc_pca_v = float(auc_pca)
                delta_auc = auc_pca_v - auc_raw_v
            else:
                auc_raw_v = float("nan")
                auc_pca_v = float("nan")
                delta_auc = float("nan")

            cv_auc_raw = model_a.cv_auc_mean
            cv_auc_pca = model_b.cv_auc_mean
            if cv_auc_raw is not None and cv_auc_pca is not None:
                cv_auc_raw_v = float(cv_auc_raw)
                cv_auc_pca_v = float(cv_auc_pca)
                delta_cv_auc = cv_auc_pca_v - cv_auc_raw_v
            else:
                cv_auc_raw_v = float("nan")
                cv_auc_pca_v = float("nan")
                delta_cv_auc = float("nan")

            rows.append(
                {
                    "Model": model_a.name,
                    "Accuracy (raw)": model_a.accuracy,
                    "Accuracy (PCA)": model_b.accuracy,
                    "ΔAccuracy": model_b.accuracy - model_a.accuracy,
                    "Precision (raw)": model_a.precision_macro,
                    "Precision (PCA)": model_b.precision_macro,
                    "ΔPrecision": model_b.precision_macro - model_a.precision_macro,
                    "Recall (raw)": model_a.recall_macro,
                    "Recall (PCA)": model_b.recall_macro,
                    "ΔRecall": model_b.recall_macro - model_a.recall_macro,
                    "F1 macro (raw)": model_a.f1_macro,
                    "F1 macro (PCA)": model_b.f1_macro,
                    "ΔF1 macro": model_b.f1_macro - model_a.f1_macro,
                    "AUC-ROC (raw)": auc_raw_v,
                    "AUC-ROC (PCA)": auc_pca_v,
                    "ΔAUC-ROC": delta_auc,
                    "CV Accuracy (raw)": model_a.cv_accuracy_mean,
                    "CV Accuracy (PCA)": model_b.cv_accuracy_mean,
                    "ΔCV Accuracy": model_b.cv_accuracy_mean - model_a.cv_accuracy_mean,
                    "CV AUC-ROC (raw)": cv_auc_raw_v,
                    "CV AUC-ROC (PCA)": cv_auc_pca_v,
                    "ΔCV AUC-ROC": delta_cv_auc,
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
        lines.append("=" * 130)
        lines.append("MODEL COMPARISON — RAW FEATURES vs PCA COMPONENTS")
        lines.append("=" * 130)
        lines.append(f"Raw features : {len(self.report_a.feature_names)}")
        lines.append(f"PCA features : {len(self.report_b.feature_names)}")
        lines.append("")

        header = (
            f"{'Model':<28}"
            f"{'Acc raw':>9}{'Acc pca':>9}{'ΔAcc':>8}"
            f"{'Prec raw':>9}{'Prec pca':>9}{'ΔPrec':>8}"
            f"{'Rec raw':>9}{'Rec pca':>9}{'ΔRec':>8}"
            f"{'F1 raw':>9}{'F1 pca':>9}{'ΔF1':>8}"
            f"{'AUC raw':>9}{'AUC pca':>9}{'ΔAUC':>8}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in self.rows:
            lines.append(
                f"{row['Model']:<28}"
                f"{row['Accuracy (raw)']:>9.4f}"
                f"{row['Accuracy (PCA)']:>9.4f}"
                f"{row['ΔAccuracy']:>+8.4f}"
                f"{row['Precision (raw)']:>9.4f}"
                f"{row['Precision (PCA)']:>9.4f}"
                f"{row['ΔPrecision']:>+8.4f}"
                f"{row['Recall (raw)']:>9.4f}"
                f"{row['Recall (PCA)']:>9.4f}"
                f"{row['ΔRecall']:>+8.4f}"
                f"{row['F1 macro (raw)']:>9.4f}"
                f"{row['F1 macro (PCA)']:>9.4f}"
                f"{row['ΔF1 macro']:>+8.4f}"
                f"{row['AUC-ROC (raw)']:>9.4f}"
                f"{row['AUC-ROC (PCA)']:>9.4f}"
                f"{row['ΔAUC-ROC']:>+8.4f}"
            )

        lines.append("")
        lines.append("Cross-validated comparison:")
        header_cv = (
            f"{'Model':<28}"
            f"{'CV Acc raw':>12}{'CV Acc pca':>12}{'ΔCV Acc':>10}"
            f"{'CV AUC raw':>12}{'CV AUC pca':>12}{'ΔCV AUC':>10}"
        )
        lines.append(header_cv)
        lines.append("-" * len(header_cv))
        for row in self.rows:
            lines.append(
                f"{row['Model']:<28}"
                f"{row['CV Accuracy (raw)']:>12.4f}"
                f"{row['CV Accuracy (PCA)']:>12.4f}"
                f"{row['ΔCV Accuracy']:>+10.4f}"
                f"{row['CV AUC-ROC (raw)']:>12.4f}"
                f"{row['CV AUC-ROC (PCA)']:>12.4f}"
                f"{row['ΔCV AUC-ROC']:>+10.4f}"
            )

        if self.not_applicable:
            lines.append("")
            lines.append("Not applicable on PCA (count-based NB):")
            for name in self.not_applicable:
                lines.append(f"  - {name}")

        lines.append("")
        lines.append("Winners per metric (test set):")
        for metric, label in METRIC_LABELS.items():
            best_raw = max(
                (m for m in self.report_a.models if m.name not in NOT_APPLICABLE_ON_PCA),
                key=lambda m: getattr(m, metric),
            )
            best_pca = max(
                (m for m in self.report_b.models if m.name not in NOT_APPLICABLE_ON_PCA),
                key=lambda m: getattr(m, metric),
            )
            raw_value = getattr(best_raw, metric)
            pca_value = getattr(best_pca, metric)
            overall = "RAW" if raw_value >= pca_value else "PCA"
            lines.append(
                f"  {label:<22}  RAW: {best_raw.name} = {raw_value:.4f} | "
                f"PCA: {best_pca.name} = {pca_value:.4f}  → winner: {overall}"
            )

        lines.append("")
        best_raw = self.best("raw", "accuracy")
        best_pca = self.best("pca", "accuracy")
        lines.append(
            f"Best overall on test accuracy — RAW: {best_raw.name} "
            f"({best_raw.accuracy:.4f})  vs  PCA: {best_pca.name} "
            f"({best_pca.accuracy:.4f})"
        )
        lines.append("=" * 130)
        return "\n".join(lines)


def save_comparison_csv(comparison, path, out_dir="comp"):
    os.makedirs(out_dir, exist_ok=True)
    comparison.dataframe().to_csv(path, index=False)
    return path