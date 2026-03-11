"""
Explainability Module — SHAP-style manual implementation
(No external SHAP library required — uses permutation & tree-based methods)
Breast Cancer Global ML Project
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")


PINK = "#C0185B"
PINK_LIGHT = "#F48FB1"
PINK_MID = "#E91E8C"
GREEN = "#2E7D32"
BLUE = "#1565C0"
ORANGE = "#E65100"
GRAY = "#546E7A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ──────────────────────────────────────────────────────────────────────────────
#  1. Feature Importance Comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_importance_comparison(models_dict, X, y, feature_names, top_n=15, save_path=None):
    """Plot permutation importance for multiple models side by side"""
    importances = {}
    for name, model in models_dict.items():
        try:
            model.fit(X, y)
            r = permutation_importance(model, X, y, n_repeats=30, random_state=42, n_jobs=-1)
            importances[name] = r.importances_mean
        except Exception:
            pass

    if not importances:
        return

    # Average importance across models
    avg_imp = np.mean(list(importances.values()), axis=0)
    top_idx = np.argsort(avg_imp)[-top_n:]

    fig, axes = plt.subplots(1, len(importances), figsize=(5 * len(importances), 8))
    if len(importances) == 1:
        axes = [axes]

    for ax, (name, imp) in zip(axes, importances.items()):
        imp_top = imp[top_idx]
        feat_top = [feature_names[i] for i in top_idx]
        colors = [PINK if v > np.median(imp_top) else PINK_LIGHT for v in imp_top]
        bars = ax.barh(feat_top, imp_top, color=colors, edgecolor="white", height=0.7)
        ax.set_title(name, fontsize=11, fontweight="bold", color=PINK)
        ax.set_xlabel("Permutation Importance", fontsize=9)
        ax.axvline(0, color=GRAY, linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, imp_top):
            if val > 0:
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=7.5)

    fig.suptitle("Feature Importance Comparison Across Models", fontsize=14,
                 fontweight="bold", color=PINK, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return importances


# ──────────────────────────────────────────────────────────────────────────────
#  2. Manual SHAP-style Waterfall (via marginal contribution)
# ──────────────────────────────────────────────────────────────────────────────

def compute_manual_shap(model, X, X_baseline, feature_names, sample_idx):
    """Approximate SHAP values via marginal contribution sampling"""
    np.random.seed(42)
    n_features = X.shape[1]
    n_samples = min(50, X.shape[0])
    shap_vals = np.zeros(n_features)
    x_instance = X[sample_idx].copy()

    for feat_idx in range(n_features):
        gains = []
        for _ in range(n_samples):
            bg_idx = np.random.randint(0, X.shape[0])
            bg = X[bg_idx].copy()

            # with feature
            x_with = bg.copy()
            x_with[feat_idx] = x_instance[feat_idx]
            # without feature
            x_without = bg.copy()

            try:
                pred_with    = model.predict(x_with.reshape(1, -1))[0]
                pred_without = model.predict(x_without.reshape(1, -1))[0]
                gains.append(pred_with - pred_without)
            except Exception:
                pass
        shap_vals[feat_idx] = np.mean(gains) if gains else 0.0

    return shap_vals


def plot_waterfall(shap_vals, feature_names, country_name, baseline, prediction, save_path=None):
    """SHAP-style waterfall plot for a single country"""
    n = len(shap_vals)
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:12]
    vals = shap_vals[sorted_idx]
    names = [feature_names[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 7))
    running = baseline
    positions = []
    bottoms = []

    for i, (v, nm) in enumerate(zip(vals[::-1], names[::-1])):
        color = GREEN if v > 0 else PINK
        ax.barh(i, v, left=running, color=color, height=0.6,
                edgecolor="white", linewidth=0.5, alpha=0.88)
        ax.text(running + v + (0.3 if v > 0 else -0.3), i,
                f"{v:+.2f}", va="center", ha="left" if v > 0 else "right",
                fontsize=8.5, color=color, fontweight="bold")
        ax.text(-2, i, nm, va="center", ha="right", fontsize=9, color="#333")
        running += v

    ax.axvline(baseline, color=GRAY, linewidth=1.2, linestyle="--", alpha=0.7, label=f"Baseline: {baseline:.1f}%")
    ax.axvline(prediction, color=PINK, linewidth=2, linestyle="-", alpha=0.9, label=f"Prediction: {prediction:.1f}%")

    ax.set_yticks([])
    ax.set_xlabel("5-Year Survival % Contribution", fontsize=10)
    ax.set_title(f"Feature Contribution — {country_name}", fontsize=13,
                 fontweight="bold", color=PINK, pad=15)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(baseline - 25, baseline + 35)

    # Color legend
    pos_patch = mpatches.Patch(color=GREEN, label="↑ Increases Survival")
    neg_patch = mpatches.Patch(color=PINK, label="↓ Decreases Survival")
    ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
#  3. Partial Dependence Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_pdp_top_features(model, X, feature_names, top_features_idx, target_name, save_path=None):
    """Custom PDP for top features"""
    n_feat = min(len(top_features_idx), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for plot_i, feat_idx in enumerate(top_features_idx[:n_feat]):
        ax = axes[plot_i]
        feat_vals = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 60)
        pdp_vals = []
        for fv in feat_vals:
            X_mod = X.copy()
            X_mod[:, feat_idx] = fv
            pdp_vals.append(model.predict(X_mod).mean())

        ax.plot(feat_vals, pdp_vals, color=PINK, linewidth=2.5)
        ax.fill_between(feat_vals, pdp_vals,
                        alpha=0.12, color=PINK_LIGHT)
        ax.axvline(X[:, feat_idx].mean(), color=GRAY, linewidth=1, linestyle="--",
                   label=f"Mean = {X[:, feat_idx].mean():.1f}")
        ax.set_xlabel(feature_names[feat_idx], fontsize=9)
        ax.set_ylabel(target_name, fontsize=9)
        ax.set_title(f"PDP: {feature_names[feat_idx]}", fontsize=10,
                     fontweight="bold", color=PINK)
        ax.legend(fontsize=7.5)
        ax.tick_params(labelsize=8)

    for ax in axes[n_feat:]:
        ax.set_visible(False)

    fig.suptitle(f"Partial Dependence Plots — {target_name}", fontsize=14,
                 fontweight="bold", color=PINK, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
#  4. Survival Gap Decomposition
# ──────────────────────────────────────────────────────────────────────────────

def plot_survival_gap_analysis(df, feature_names_short, save_path=None):
    """Bar chart showing mean survival gap per income group + key drivers"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: survival gap by income group
    gap_data = df.groupby("Income_Group")["Survival_Gap_vs_Benchmark"].agg(["mean", "std"])
    income_order = ["Low", "LowerMid", "UpperMid", "High"]
    gap_data = gap_data.reindex([g for g in income_order if g in gap_data.index])
    colors_bar = [PINK if v < 0 else GREEN for v in gap_data["mean"]]

    axes[0].barh(gap_data.index, gap_data["mean"], xerr=gap_data["std"],
                 color=colors_bar, alpha=0.85, edgecolor="white", height=0.5,
                 error_kw={"capsize": 4, "ecolor": GRAY, "capthick": 1.5})
    axes[0].axvline(0, color=GRAY, linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Survival Gap vs Income Benchmark (%)", fontsize=10)
    axes[0].set_title("Survival Gap by Income Group", fontsize=12,
                      fontweight="bold", color=PINK)
    for i, (idx, row) in enumerate(gap_data.iterrows()):
        axes[0].text(row["mean"] + (0.5 if row["mean"] >= 0 else -0.5), i,
                     f"{row['mean']:+.1f}%", va="center",
                     ha="left" if row["mean"] >= 0 else "right",
                     fontsize=9, fontweight="bold", color=GRAY)

    # Right: Correlation of key features with survival gap
    key_features = ["Mammography_Coverage_Pct", "Treatment_Access_Score",
                    "Stage_I_II_Pct", "Healthcare_Quality_Index",
                    "Screening_Program", "Detection_Efficiency_Score"]
    corrs = []
    feat_labels = []
    for feat in key_features:
        if feat in df.columns:
            corr = df[feat].astype(float).corr(df["Survival_Gap_vs_Benchmark"])
            corrs.append(corr)
            feat_labels.append(feat.replace("_", "\n"))

    colors_c = [GREEN if c > 0 else PINK for c in corrs]
    axes[1].barh(feat_labels, corrs, color=colors_c, alpha=0.85,
                 edgecolor="white", height=0.55)
    axes[1].axvline(0, color=GRAY, linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Pearson Correlation with Survival Gap", fontsize=10)
    axes[1].set_title("Feature Correlation with Survival Gap", fontsize=12,
                      fontweight="bold", color=PINK)
    for i, (c, lbl) in enumerate(zip(corrs, feat_labels)):
        axes[1].text(c + (0.02 if c > 0 else -0.02), i,
                     f"r={c:.2f}", va="center",
                     ha="left" if c > 0 else "right", fontsize=8.5, color=GRAY)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
#  5. Counterfactual Simulation
# ──────────────────────────────────────────────────────────────────────────────

def counterfactual_analysis(model, X, feature_names, country_idx, country_name,
                             target_survival, current_survival, feature_ranges):
    """
    Simple 1-feature counterfactual: for each feature, find the value needed
    to close the gap to target_survival.
    """
    results = []
    x_base = X[country_idx].copy()

    for feat_name, (feat_min, feat_max) in feature_ranges.items():
        if feat_name not in feature_names:
            continue
        feat_idx = feature_names.index(feat_name)
        test_vals = np.linspace(feat_min, feat_max, 100)
        best_val = None
        best_pred = current_survival

        for tv in test_vals:
            x_test = x_base.copy()
            x_test[feat_idx] = tv
            pred = model.predict(x_test.reshape(1, -1))[0]
            if abs(pred - target_survival) < abs(best_pred - target_survival):
                best_val = tv
                best_pred = pred

        if best_val is not None:
            delta_feat = best_val - x_base[feat_idx]
            delta_surv = best_pred - current_survival
            results.append({
                "Feature": feat_name,
                "Current Value": round(x_base[feat_idx], 2),
                "Required Value": round(best_val, 2),
                "Change Needed": round(delta_feat, 2),
                "Predicted Survival": round(best_pred, 2),
                "Survival Gain": round(delta_surv, 2),
            })

    return pd.DataFrame(results).sort_values("Survival Gain", ascending=False)
