import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools.tools import add_constant


class HeteroscedasticityResult:
    def __init__(self, model_name: str, residuals: np.ndarray, fitted: np.ndarray):
        self.model_name = model_name
        self.residuals = np.asarray(residuals).ravel()
        self.fitted = np.asarray(fitted).ravel()

        n = len(self.residuals)
        self.n = n

        self.mean_residual = float(np.mean(self.residuals))
        self.std_residual = float(np.std(self.residuals, ddof=1))

        self.spearman_rho, self.spearman_p = stats.spearmanr(
            self.fitted, np.abs(self.residuals)
        )
        self.spearman_rho = float(self.spearman_rho)
        self.spearman_p = float(self.spearman_p)

        self.pearson_sq_rho, self.pearson_sq_p = stats.pearsonr(
            self.fitted, self.residuals**2
        )
        self.pearson_sq_rho = float(self.pearson_sq_rho)
        self.pearson_sq_p = float(self.pearson_sq_p)

        design = add_constant(self.fitted.reshape(-1, 1))
        bp_lm, bp_p, bp_f, bp_f_p = het_breuschpagan(self.residuals, design)
        self.bp_lm = float(bp_lm)
        self.bp_p = float(bp_p)
        self.bp_f = float(bp_f)
        self.bp_f_p = float(bp_f_p)

        self.goldfeld_quandt_f, self.goldfeld_quandt_p = self._goldfeld_quandt()

    def _goldfeld_quandt(self, drop_fraction: float = 0.2):
        order = np.argsort(self.fitted)
        sorted_resid = self.residuals[order]
        n = len(sorted_resid)
        drop = int(n * drop_fraction)
        low = sorted_resid[: n - drop]
        high = sorted_resid[drop:]
        if len(low) < 3 or len(high) < 3:
            return float("nan"), float("nan")
        var_low = np.var(low, ddof=1)
        var_high = np.var(high, ddof=1)
        if var_low == 0 or var_high == 0:
            return float("nan"), float("nan")
        f_stat = var_high / var_low if var_high >= var_low else var_low / var_high
        df1 = len(high) - 1
        df2 = len(low) - 1
        p_value = 2 * min(
            stats.f.cdf(f_stat, df1, df2),
            1 - stats.f.cdf(f_stat, df1, df2),
        )
        return float(f_stat), float(p_value)

    @property
    def verdict(self) -> str:
        alpha = 0.05
        evidence = []
        if self.bp_p < alpha:
            evidence.append("Breusch-Pagan")
        if self.goldfeld_quandt_p < alpha:
            evidence.append("Goldfeld-Quandt")
        if self.spearman_p < alpha:
            evidence.append("Spearman (|residual| vs fitted)")

        if evidence:
            return (
                "HETEROSCEDASTICITY detected — significant evidence from: "
                + ", ".join(evidence)
                + "."
            )
        return (
            "HOMOSCEDASTICITY not rejected — no significant evidence of "
            "non-constant variance."
        )

    def __str__(self):
        lines = []
        lines.append("-" * 60)
        lines.append(f"Heteroscedasticity analysis: {self.model_name}")
        lines.append("-" * 60)
        lines.append(f"Observations        : {self.n}")
        lines.append(f"Mean residual       : {self.mean_residual:,.2f}")
        lines.append(f"Std. residual       : {self.std_residual:,.2f}")
        lines.append("")
        lines.append("Breusch-Pagan test (residuals vs fitted):")
        lines.append(f"  LM statistic      : {self.bp_lm:,.4f}")
        lines.append(f"  LM p-value        : {self.bp_p:.6f}")
        lines.append(f"  F statistic       : {self.bp_f:,.4f}")
        lines.append(f"  F p-value         : {self.bp_f_p:.6f}")
        lines.append("")
        lines.append("Goldfeld-Quandt test (variance ratio, low vs high fitted):")
        lines.append(f"  F statistic       : {self.goldfeld_quandt_f:,.4f}")
        lines.append(f"  p-value           : {self.goldfeld_quandt_p:.6f}")
        lines.append("")
        lines.append("Spearman correlation (|residual| vs fitted):")
        lines.append(f"  rho               : {self.spearman_rho:,.4f}")
        lines.append(f"  p-value           : {self.spearman_p:.6f}")
        lines.append("")
        lines.append("Pearson correlation (residual² vs fitted):")
        lines.append(f"  r                 : {self.pearson_sq_rho:,.4f}")
        lines.append(f"  p-value           : {self.pearson_sq_p:.6f}")
        lines.append("")
        lines.append(f"Verdict             : {self.verdict}")
        return "\n".join(lines)


class HeteroscedasticityReport:
    def __init__(self, results: list):
        self.results = results

    def __str__(self):
        lines = []
        lines.append("=" * 60)
        lines.append("HETEROSCEDASTICITY REPORT")
        lines.append("=" * 60)
        for result in self.results:
            lines.append(str(result))
            lines.append("")
        return "\n".join(lines)


def analyse_heteroscedasticity(models: list) -> HeteroscedasticityReport:
    results = []
    for model in models:
        if model.residuals is None or model.y_pred is None:
            continue
        results.append(
            HeteroscedasticityResult(
                model_name=model.name,
                residuals=model.residuals,
                fitted=model.y_pred,
            )
        )
    return HeteroscedasticityReport(results)
