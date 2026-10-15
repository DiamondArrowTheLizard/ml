import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


class VIFResult:
    def __init__(self, feature_names, vif_values):
        self.feature_names = list(feature_names)
        self.vif_values = list(vif_values)
        self.series = pd.Series(self.vif_values, index=self.feature_names)

    def max_vif(self):
        return float(max(self.vif_values)) if self.vif_values else float("nan")

    def high_vif_features(self, threshold: float = 10.0):
        return [name for name, value in self.series.items() if value > threshold]

    def __str__(self):
        lines = ["Variance Inflation Factors:"]
        for name, value in self.series.items():
            flag = "  <-- high" if value > 10.0 else ""
            lines.append(f"  {name}: {value:,.3f}{flag}")
        lines.append(f"Max VIF: {self.max_vif():,.3f}")
        return "\n".join(lines)


class PCAResult:
    def __init__(
        self,
        feature_names: list,
        variance_threshold: float,
        max_components: int,
        vif_before: VIFResult,
        pca: PCA,
        scaler: StandardScaler,
        X_pca: np.ndarray,
        y: pd.Series,
        components_df: pd.DataFrame,
        loadings_df: pd.DataFrame,
        vif_after: VIFResult,
    ):
        self.feature_names = feature_names
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self.vif_before = vif_before
        self.pca = pca
        self.scaler = scaler
        self.X_pca = X_pca
        self.y = y
        self.components_df = components_df
        self.loadings_df = loadings_df
        self.vif_after = vif_after

    @property
    def n_components(self) -> int:
        return int(self.pca.n_components_)

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    @property
    def cumulative_variance_ratio(self) -> np.ndarray:
        return np.cumsum(self.explained_variance_ratio)

    def component_names(self) -> list:
        return [f"PC{i + 1}" for i in range(self.n_components)]

    def top_loadings(self, component: str, n: int = 5) -> pd.Series:
        series = self.loadings_df[component].sort_values(key=np.abs, ascending=False)
        return series.head(n)

    def __str__(self):
        lines = []
        lines.append("=" * 60)
        lines.append("PCA ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"Original features : {len(self.feature_names)}")
        lines.append(f"Max components    : {self.max_components}")
        lines.append(
            f"Variance threshold: {self.variance_threshold:.2%} "
            f"(components kept: {self.n_components})"
        )
        lines.append("")
        lines.append("Multicollinearity check (VIF) BEFORE PCA (first 20 features):")
        vif_head = self.vif_before.series.head(20)
        for name, value in vif_head.items():
            flag = "  <-- high" if value > 10.0 else ""
            lines.append(f"  {name}: {value:,.3f}{flag}")
        lines.append(
            f"  ... ({len(self.vif_before.series)} features total, "
            f"max VIF = {self.vif_before.max_vif():,.3f})"
        )
        lines.append("")
        lines.append("Explained variance per component:")
        for i, ratio in enumerate(self.explained_variance_ratio):
            cum = self.cumulative_variance_ratio[i]
            lines.append(
                f"  PC{i + 1}: {ratio:.4f} ({ratio:.2%})  cumulative: {cum:.4f}"
            )
        lines.append("")
        lines.append("Top contributing features per component (first 5 components):")
        for component in self.component_names()[:5]:
            lines.append(f"  {component}:")
            for feature, value in self.top_loadings(component, n=5).items():
                lines.append(f"    {feature}: {value:,.4f}")
        lines.append("")
        lines.append("Multicollinearity check (VIF) AFTER PCA:")
        lines.append(str(self.vif_after))
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


def _compute_vif(X: np.ndarray, feature_names) -> VIFResult:
    X_with_const = np.column_stack([np.ones(len(X)), X])
    values = []
    for i in range(1, X_with_const.shape[1]):
        values.append(variance_inflation_factor(X_with_const, i))
    return VIFResult(feature_names, values)


def run_pca_analysis(
    X: pd.DataFrame,
    y: pd.Series,
    variance_threshold: float = 0.95,
    max_components: int = 20,
    compute_vif_before: bool = True,
) -> PCAResult:

    feature_names = list(X.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values.astype(float))

    if compute_vif_before:
        vif_before = _compute_vif(X_scaled, feature_names)
    else:
        vif_before = VIFResult(feature_names, [float("nan")] * len(feature_names))

    n_features = X_scaled.shape[1]
    upper_bound = min(max_components, n_features)

    pca = PCA(
        n_components=upper_bound,
        svd_solver="randomized",
        random_state=42,
    )
    X_pca_full = pca.fit_transform(X_scaled)

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumulative, variance_threshold) + 1)
    n_keep = max(1, min(n_keep, upper_bound))

    X_pca = X_pca_full[:, :n_keep]
    component_names = [f"PC{i + 1}" for i in range(n_keep)]
    components_df = pd.DataFrame(X_pca, columns=component_names)

    loadings_df = pd.DataFrame(
        pca.components_[:n_keep].T,
        index=feature_names,
        columns=component_names,
    )

    vif_after = _compute_vif(X_pca, component_names)

    trimmed_pca = PCA(
        n_components=n_keep,
        svd_solver="randomized",
        random_state=42,
    ).fit(X_scaled)

    return PCAResult(
        feature_names=feature_names,
        variance_threshold=variance_threshold,
        max_components=max_components,
        vif_before=vif_before,
        pca=trimmed_pca,
        scaler=scaler,
        X_pca=X_pca,
        y=y,
        components_df=components_df,
        loadings_df=loadings_df,
        vif_after=vif_after,
    )
