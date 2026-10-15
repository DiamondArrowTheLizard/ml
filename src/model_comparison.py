import os

import pandas as pd

METRIC_LABELS = {
    "rmse": "RMSE",
    "r2": "R²",
    "mape": "MAPE",
}

LOWER_IS_BETTER = {"rmse": True, "r2": False, "mape": True}


class ModelComparison:
    def __init__(self, report_a, report_b):
        self.report_a = report_a
        self.report_b = report_b
        self.rows = self._build_rows()

    def _build_rows(self):
        rows = []
        for model_a in self.report_a.models:
            model_b = next(
                (m for m in self.report_b.models if m.name == model_a.name),
                None,
            )
            if model_b is None:
                continue
            rows.append(
                {
                    "Model": model_a.name,
                    "RMSE (raw)": model_a.rmse,
                    "RMSE (PCA)": model_b.rmse,
                    "ΔRMSE": model_b.rmse - model_a.rmse,
                    "R² (raw)": model_a.r2,
                    "R² (PCA)": model_b.r2,
                    "ΔR²": model_b.r2 - model_a.r2,
                    "MAPE (raw)": model_a.mape,
                    "MAPE (PCA)": model_b.mape,
                    "ΔMAPE": model_b.mape - model_a.mape,
                    "CV RMSE (raw)": model_a.cv_rmse_mean,
                    "CV RMSE (PCA)": model_b.cv_rmse_mean,
                    "ΔCV RMSE": model_b.cv_rmse_mean - model_a.cv_rmse_mean,
                    "CV R² (raw)": model_a.cv_r2_mean,
                    "CV R² (PCA)": model_b.cv_r2_mean,
                    "ΔCV R²": model_b.cv_r2_mean - model_a.cv_r2_mean,
                }
            )
        return rows

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def best(self, label: str, metric: str = "rmse"):
        report = self.report_a if label == "raw" else self.report_b
        return report.best_model(metric)

    def per_metric_winners(self):
        winners = {}
        for metric in METRIC_LABELS:
            lower_better = LOWER_IS_BETTER[metric]
            best_raw = sorted(
                self.report_a.models,
                key=lambda m: getattr(m, metric),
                reverse=not lower_better,
            )[0]
            best_pca = sorted(
                self.report_b.models,
                key=lambda m: getattr(m, metric),
                reverse=not lower_better,
            )[0]
            winners[metric] = {
                "raw": (best_raw.name, getattr(best_raw, metric)),
                "pca": (best_pca.name, getattr(best_pca, metric)),
            }
        return winners

    def __str__(self):
        lines = []
        lines.append("=" * 90)
        lines.append("MODEL COMPARISON — RAW FEATURES vs PCA COMPONENTS")
        lines.append("=" * 90)
        lines.append(
            f"Raw features : {len(self.report_a.feature_names)} features "
            f"({self.report_a.label or 'raw'})"
        )
        lines.append(
            f"PCA features : {len(self.report_b.feature_names)} features "
            f"({self.report_b.label or 'pca'})"
        )
        lines.append("")

        self.dataframe()
        header = (
            f"{'Model':<20}"
            f"{'RMSE raw':>12}{'RMSE pca':>12}{'ΔRMSE':>12}"
            f"{'R² raw':>10}{'R² pca':>10}{'ΔR²':>10}"
            f"{'MAPE raw':>11}{'MAPE pca':>11}{'ΔMAPE':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in self.rows:
            lines.append(
                f"{row['Model']:<20}"
                f"{row['RMSE (raw)']:>12,.2f}"
                f"{row['RMSE (PCA)']:>12,.2f}"
                f"{row['ΔRMSE']:>+12,.2f}"
                f"{row['R² (raw)']:>10.4f}"
                f"{row['R² (PCA)']:>10.4f}"
                f"{row['ΔR²']:>+10.4f}"
                f"{row['MAPE (raw)']:>11.4f}"
                f"{row['MAPE (PCA)']:>11.4f}"
                f"{row['ΔMAPE']:>+10.4f}"
            )

        lines.append("")
        lines.append("Cross-validated comparison:")
        header_cv = (
            f"{'Model':<20}"
            f"{'CV RMSE raw':>14}{'CV RMSE pca':>14}{'ΔCV RMSE':>14}"
            f"{'CV R² raw':>12}{'CV R² pca':>12}{'ΔCV R²':>12}"
        )
        lines.append(header_cv)
        lines.append("-" * len(header_cv))
        for row in self.rows:
            lines.append(
                f"{row['Model']:<20}"
                f"{row['CV RMSE (raw)']:>14,.2f}"
                f"{row['CV RMSE (PCA)']:>14,.2f}"
                f"{row['ΔCV RMSE']:>+14,.2f}"
                f"{row['CV R² (raw)']:>12.4f}"
                f"{row['CV R² (PCA)']:>12.4f}"
                f"{row['ΔCV R²']:>+12.4f}"
            )

        lines.append("")
        lines.append("Winners per metric (test set):")
        for metric, label in METRIC_LABELS.items():
            winners = self.per_metric_winners()[metric]
            raw_name, raw_value = winners["raw"]
            pca_name, pca_value = winners["pca"]
            lower_better = LOWER_IS_BETTER[metric]
            if lower_better:
                overall = "RAW" if raw_value < pca_value else "PCA"
            else:
                overall = "RAW" if raw_value > pca_value else "PCA"
            lines.append(
                f"  {label:<5}  RAW: {raw_name} = {raw_value:,.4f} | "
                f"PCA: {pca_name} = {pca_value:,.4f}  →  winner: {overall}"
            )

        lines.append("")
        best_raw = self.best("raw", "rmse")
        best_pca = self.best("pca", "rmse")
        lines.append(
            f"Best overall on test RMSE — RAW: {best_raw.name} "
            f"({best_raw.rmse:,.2f})  vs  PCA: {best_pca.name} "
            f"({best_pca.rmse:,.2f})"
        )
        lines.append("=" * 90)
        return "\n".join(lines)


def save_comparison_csv(comparison: ModelComparison, path: str, out_dir="comp"):
    os.makedirs(out_dir, exist_ok=True)
    comparison.dataframe().to_csv(path, index=False)
    return path
