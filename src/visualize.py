"""
Visualization Module — EDA & Model Results
Breast Cancer Global ML Project
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

PINK   = "#C0185B"
PMID   = "#E91E8C"
PLIGHT = "#F48FB1"
PLIGHTEST = "#FCE4EC"
GREEN  = "#2E7D32"
GLIGHT = "#A5D6A7"
BLUE   = "#1565C0"
BLIGHT = "#90CAF9"
ORANGE = "#E65100"
OLIGHT = "#FFCC80"
GRAY   = "#546E7A"
DGRAY  = "#37474F"

INCOME_COLORS = {"Low": ORANGE, "LowerMid": PMID, "UpperMid": BLUE, "High": GREEN}
TIER_COLORS   = {"High": PINK, "Mid": ORANGE, "Low": GREEN}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": PLIGHT,
})


# ──────────────────────────────────────────────────────────────────────────────
#  EDA — Overview Dashboard
# ──────────────────────────────────────────────────────────────────────────────

def plot_eda_overview(df, save_path=None):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("white")
    gs = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

    # 1. Survival Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["Five_Year_Survival_Pct"], bins=15, color=PINK, edgecolor="white",
             alpha=0.85, linewidth=1.2)
    ax1.axvline(df["Five_Year_Survival_Pct"].mean(), color=DGRAY,
                linewidth=2, linestyle="--", label=f"Mean={df['Five_Year_Survival_Pct'].mean():.1f}%")
    ax1.axvline(60, color=BLUE, linewidth=1.5, linestyle=":", label="WHO 60% target")
    ax1.set_title("5-Year Survival Distribution", fontweight="bold", color=PINK, fontsize=10)
    ax1.set_xlabel("Survival (%)")
    ax1.legend(fontsize=7.5)

    # 2. Stage I/II distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df["Stage_I_II_Pct"], bins=12, color=PMID, edgecolor="white", alpha=0.85)
    ax2.axvline(60, color=BLUE, linewidth=2, linestyle="--", label="WHO 60% target")
    ax2.axvline(df["Stage_I_II_Pct"].mean(), color=ORANGE, linewidth=1.5,
                linestyle=":", label=f"Mean={df['Stage_I_II_Pct'].mean():.1f}%")
    ax2.set_title("Stage I/II % Distribution", fontweight="bold", color=PMID, fontsize=10)
    ax2.set_xlabel("Stage I/II (%)")
    ax2.legend(fontsize=7.5)

    # 3. Survival vs Mammography
    ax3 = fig.add_subplot(gs[0, 2])
    sc = ax3.scatter(df["Mammography_Coverage_Pct"], df["Five_Year_Survival_Pct"],
               c=df["Treatment_Access_Score"], cmap="RdYlGn", s=60, alpha=0.8,
               edgecolors="white", linewidth=0.5)
    cb = plt.colorbar(sc, ax=ax3)
    cb.set_label("Treatment Access", fontsize=7.5)
    ax3.set_xlabel("Mammography Coverage (%)")
    ax3.set_ylabel("5-Year Survival (%)")
    ax3.set_title("Survival vs Mammography", fontweight="bold", color=PINK, fontsize=10)
    # Add regression line
    z = np.polyfit(df["Mammography_Coverage_Pct"], df["Five_Year_Survival_Pct"], 1)
    xfit = np.linspace(df["Mammography_Coverage_Pct"].min(), df["Mammography_Coverage_Pct"].max(), 100)
    ax3.plot(xfit, np.polyval(z, xfit), color=PINK, linewidth=2, linestyle="--", alpha=0.7)

    # 4. Mortality Rate by Continent
    ax4 = fig.add_subplot(gs[0, 3])
    cont_surv = df.groupby("Continent")["Five_Year_Survival_Pct"].mean().sort_values()
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(cont_surv)))
    bars = ax4.barh(cont_surv.index, cont_surv.values, color=colors, edgecolor="white", height=0.6)
    ax4.axvline(80, color=BLUE, linewidth=1.5, linestyle="--", alpha=0.7, label="80% mark")
    ax4.set_xlabel("Mean 5-Year Survival (%)")
    ax4.set_title("Survival by Continent", fontweight="bold", color=PINK, fontsize=10)
    for bar, val in zip(bars, cont_surv.values):
        ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2, f"{val:.0f}%",
                 va="center", fontsize=8)

    # 5. Correlation heatmap (key features)
    ax5 = fig.add_subplot(gs[1, :2])
    key_cols = ["Five_Year_Survival_Pct", "Stage_I_II_Pct", "Mammography_Coverage_Pct",
                "Treatment_Access_Score", "Incidence_Rate_Per_100K", "Mortality_Rate_Per_100K",
                "Healthcare_Quality_Index", "Case_Fatality_Ratio"]
    key_cols = [c for c in key_cols if c in df.columns]
    corr = df[key_cols].corr()
    im = ax5.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax5, shrink=0.8)
    ax5.set_xticks(range(len(key_cols)))
    ax5.set_yticks(range(len(key_cols)))
    labels_short = [c.replace("_", "\n").replace("Pct", "%") for c in key_cols]
    ax5.set_xticklabels(labels_short, fontsize=7, rotation=45, ha="right")
    ax5.set_yticklabels(labels_short, fontsize=7)
    for i in range(len(key_cols)):
        for j in range(len(key_cols)):
            ax5.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                     color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    ax5.set_title("Correlation Matrix — Key Features", fontweight="bold", color=PINK, fontsize=10)

    # 6. Screening Program impact
    ax6 = fig.add_subplot(gs[1, 2])
    screen_yes = df[df["Screening_Program"] == True]["Five_Year_Survival_Pct"]
    screen_no  = df[df["Screening_Program"] == False]["Five_Year_Survival_Pct"]
    ax6.boxplot([screen_no, screen_yes], labels=["No Screening\nProgram", "Has Screening\nProgram"],
                patch_artist=True, boxprops=dict(facecolor=PLIGHT, color=PINK),
                medianprops=dict(color=PINK, linewidth=2),
                whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY))
    ax6.scatter([1]*len(screen_no),  screen_no,  color=ORANGE, alpha=0.4, s=25, zorder=5)
    ax6.scatter([2]*len(screen_yes), screen_yes, color=GREEN,  alpha=0.4, s=25, zorder=5)
    ax6.set_ylabel("5-Year Survival (%)")
    ax6.set_title("Impact of Screening Programs", fontweight="bold", color=PINK, fontsize=10)

    # 7. WHO 60-60-80 compliance
    ax7 = fig.add_subplot(gs[1, 3])
    n60 = (df["Stage_I_II_Pct"] >= 60).sum()
    n80 = (df["Treatment_Access_Score"] >= 80).sum()
    nboth = ((df["Stage_I_II_Pct"] >= 60) & (df["Treatment_Access_Score"] >= 80)).sum()
    n_total = len(df)
    who_data = {
        f"Early Detection\n≥60% (n={n60})": n60,
        f"Treatment Access\n≥80 (n={n80})": n80,
        f"Both Targets\n(n={nboth})": nboth,
        f"Neither\n(n={n_total-n60})": n_total - max(n60, n80),
    }
    colors_w = [PINK, BLUE, GREEN, ORANGE]
    wedges, texts, autotexts = ax7.pie(
        list(who_data.values()), labels=list(who_data.keys()),
        colors=colors_w, autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 7.5}
    )
    ax7.set_title("WHO 60-60-80\nCompliance (50 Countries)", fontweight="bold", color=PINK, fontsize=10)

    # 8. Incidence vs Mortality scatter
    ax8 = fig.add_subplot(gs[2, 0])
    income_colors_mapped = df["Income_Group"].map(INCOME_COLORS) if "Income_Group" in df.columns else PINK
    ax8.scatter(df["Incidence_Rate_Per_100K"], df["Mortality_Rate_Per_100K"],
                c=income_colors_mapped, s=50, alpha=0.8, edgecolors="white", linewidth=0.5)
    ax8.set_xlabel("Incidence Rate / 100K")
    ax8.set_ylabel("Mortality Rate / 100K")
    ax8.set_title("Incidence vs Mortality", fontweight="bold", color=PINK, fontsize=10)
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in INCOME_COLORS.items()]
    ax8.legend(handles=legend_patches, fontsize=7, title="Income Group")

    # 9. Treatment Access vs Survival
    ax9 = fig.add_subplot(gs[2, 1])
    ax9.scatter(df["Treatment_Access_Score"], df["Five_Year_Survival_Pct"],
                color=PINK, s=55, alpha=0.75, edgecolors=PLIGHT, linewidth=0.8)
    z2 = np.polyfit(df["Treatment_Access_Score"], df["Five_Year_Survival_Pct"], 1)
    xfit2 = np.linspace(df["Treatment_Access_Score"].min(), df["Treatment_Access_Score"].max(), 100)
    ax9.plot(xfit2, np.polyval(z2, xfit2), color=PINK, linewidth=2.5, linestyle="--")
    ax9.axhline(80, color=GREEN, linewidth=1.5, linestyle=":", alpha=0.7, label="80% survival")
    ax9.set_xlabel("Treatment Access Score")
    ax9.set_ylabel("5-Year Survival (%)")
    ax9.set_title("Treatment Access vs Survival", fontweight="bold", color=PINK, fontsize=10)
    ax9.legend(fontsize=7.5)

    # 10. Top 10 countries by survival gap
    ax10 = fig.add_subplot(gs[2, 2:])
    if "Survival_Gap_vs_Benchmark" in df.columns:
        gap_sorted = df.nsmallest(10, "Survival_Gap_vs_Benchmark")[["Country", "Survival_Gap_vs_Benchmark", "Five_Year_Survival_Pct"]]
        gap_colors = [PINK if g < -10 else ORANGE for g in gap_sorted["Survival_Gap_vs_Benchmark"]]
        bars = ax10.barh(gap_sorted["Country"], gap_sorted["Survival_Gap_vs_Benchmark"],
                         color=gap_colors, alpha=0.85, edgecolor="white", height=0.6)
        ax10.axvline(0, color=GRAY, linewidth=1.5, linestyle="--")
        for bar, (_, row) in zip(bars, gap_sorted.iterrows()):
            ax10.text(row["Survival_Gap_vs_Benchmark"] - 0.5, bar.get_y() + bar.get_height()/2,
                      f"{row['Survival_Gap_vs_Benchmark']:.1f}% (actual: {row['Five_Year_Survival_Pct']:.0f}%)",
                      va="center", ha="right", fontsize=7.5, color=DGRAY)
        ax10.set_xlabel("Survival Gap vs Income Benchmark (%)")
        ax10.set_title("Countries with Largest Survival Gap", fontweight="bold", color=PINK, fontsize=10)

    fig.suptitle("🎀 Global Breast Cancer Statistics — Exploratory Data Analysis",
                 fontsize=16, fontweight="bold", color=PINK, y=1.005)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Model Comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results_list, metric_cols, title, save_path=None):
    """Bar chart comparing models on multiple metrics"""
    df_res = pd.DataFrame(results_list).set_index("Model")
    metrics = [m for m in metric_cols if m in df_res.columns]
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = df_res[metric].sort_values(ascending=(metric in ["MAE", "RMSE", "MAPE (%)", "Max Error"]))
        ascending = metric in ["MAE", "RMSE", "MAPE (%)", "Max Error"]
        vals = vals.sort_values(ascending=ascending)
        colors = [GREEN if i == (0 if ascending else len(vals)-1) else PINK for i in range(len(vals))]
        bars = ax.barh(vals.index, vals.values, color=colors, alpha=0.82,
                       edgecolor="white", height=0.6)
        ax.set_title(metric, fontweight="bold", color=PINK, fontsize=11)
        for bar, val in zip(bars, vals.values):
            ax.text(val * 1.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8.5)
        best_label = "Lower=Better" if ascending else "Higher=Better"
        ax.text(0.98, 0.02, best_label, transform=ax.transAxes,
                ha="right", fontsize=7.5, color=GRAY, style="italic")

    fig.suptitle(title, fontsize=13, fontweight="bold", color=PINK, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Predicted vs Actual
# ──────────────────────────────────────────────────────────────────────────────

def plot_pred_vs_actual(y_true, y_pred_dict, target_name, save_path=None):
    n = len(y_pred_dict)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for i, (name, y_pred) in enumerate(y_pred_dict.items()):
        ax = axes[i]
        ax.scatter(y_true, y_pred, color=PINK, s=50, alpha=0.7,
                   edgecolors=PLIGHT, linewidth=0.5)
        lim_min = min(y_true.min(), np.array(y_pred).min()) - 2
        lim_max = max(y_true.max(), np.array(y_pred).max()) + 2
        ax.plot([lim_min, lim_max], [lim_min, lim_max], color=GRAY,
                linewidth=1.5, linestyle="--", alpha=0.7, label="Perfect Prediction")
        ax.fill_between([lim_min, lim_max],
                        [lim_min - 5, lim_max - 5],
                        [lim_min + 5, lim_max + 5],
                        alpha=0.07, color=GREEN, label="±5% band")
        from sklearn.metrics import r2_score, mean_absolute_error
        r2  = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        ax.set_title(f"{name}\nR²={r2:.3f}  MAE={mae:.2f}%", fontsize=9,
                     fontweight="bold", color=PINK)
        ax.set_xlabel(f"Actual {target_name}")
        ax.set_ylabel(f"Predicted {target_name}")
        ax.legend(fontsize=7.5)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Predicted vs Actual — {target_name} (LOOCV)", fontsize=13,
                 fontweight="bold", color=PINK, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  Clustering Visualization (PCA 2D)
# ──────────────────────────────────────────────────────────────────────────────

def plot_clusters_pca(df, X_scaled, cluster_col, label_col=None, save_path=None):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, (ax, col, title_suffix) in enumerate(zip(axes,
            [cluster_col, label_col or cluster_col],
            ["K-Means Clusters", "Risk Tier Labels"])):
        if col not in df.columns:
            col = cluster_col
        unique_vals = df[col].unique()

        if col == label_col and label_col in df.columns:
            color_map = TIER_COLORS
            c_vals = [color_map.get(str(v), GRAY) for v in df[col]]
        else:
            cmap = plt.cm.Set1
            c_vals = [cmap(i / max(len(unique_vals) - 1, 1))
                      for i, v in enumerate(df[col])]

        sizes = df["New_Cases_2022"] / df["New_Cases_2022"].max() * 400 + 30

        sc = ax.scatter(coords[:, 0], coords[:, 1], c=c_vals,
                        s=sizes, alpha=0.8, edgecolors="white", linewidth=0.7)

        for i, country in enumerate(df["Country"]):
            if df["New_Cases_2022"].iloc[i] > 50000 or \
               (col == label_col and str(df[col].iloc[i]) == "High"):
                ax.annotate(country[:12], (coords[i, 0], coords[i, 1]),
                           fontsize=6.5, ha="center", va="bottom",
                           xytext=(0, 5), textcoords="offset points",
                           color=DGRAY, fontweight="bold")

        ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=9)
        ax.set_title(f"Country Clusters — {title_suffix}", fontsize=11,
                     fontweight="bold", color=PINK)

        if col == label_col:
            patches = [mpatches.Patch(color=v, label=f"{k} Risk")
                       for k, v in TIER_COLORS.items() if k in df[col].values]
            ax.legend(handles=patches, fontsize=9, title="Risk Tier")

    fig.suptitle("PCA 2D Country Clustering — Breast Cancer Risk Tiers",
                 fontsize=14, fontweight="bold", color=PINK)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
#  WHO Policy Dashboard
# ──────────────────────────────────────────────────────────────────────────────

def plot_who_policy_dashboard(df, save_path=None):
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor("white")
    gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # 1. WHO 60-60-80 Gap by Country (top/bottom 10)
    ax1 = fig.add_subplot(gs[0, :2])
    df_sorted = df.sort_values("Stage_I_II_Pct")
    bottom10 = df_sorted.head(10)
    top10    = df_sorted.tail(10)
    combined = pd.concat([bottom10, top10])
    colors_gap = [PINK if v < 60 else GREEN for v in combined["Stage_I_II_Pct"]]
    ax1.barh(combined["Country"], combined["Stage_I_II_Pct"], color=colors_gap,
             alpha=0.83, edgecolor="white", height=0.65)
    ax1.axvline(60, color=BLUE, linewidth=2.5, linestyle="--",
                label="WHO 60% Target")
    ax1.set_xlabel("Stage I/II Diagnosis Rate (%)")
    ax1.set_title("WHO Early Detection Target (60%): 10 Worst & 10 Best Countries",
                  fontweight="bold", color=PINK, fontsize=10)
    ax1.legend(fontsize=9)
    for i, (_, row) in enumerate(combined.iterrows()):
        gap = row["Stage_I_II_Pct"] - 60
        ax1.text(row["Stage_I_II_Pct"] + 0.5, i,
                 f"{row['Stage_I_II_Pct']:.0f}% ({gap:+.0f}%)",
                 va="center", fontsize=7.5, color=DGRAY)

    # 2. Quadrant plot: Risk Matrix
    ax2 = fig.add_subplot(gs[0, 2])
    if "Risk_Tier_Cluster" in df.columns:
        tier_col = "Risk_Tier_Cluster"
    elif "Risk_Tier" in df.columns:
        tier_col = "Risk_Tier"
    else:
        tier_col = None

    if tier_col:
        tier_counts = df[tier_col].value_counts()
        colors_pie = [TIER_COLORS.get(str(k), GRAY) for k in tier_counts.index]
        wedges, texts, autotexts = ax2.pie(
            tier_counts.values, labels=[f"{k}\n(n={v})" for k, v in tier_counts.items()],
            colors=colors_pie, autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 9}
        )
        ax2.set_title("Country Risk Tier Distribution", fontweight="bold", color=PINK, fontsize=10)

    # 3. Treatment Access vs Survival (income color)
    ax3 = fig.add_subplot(gs[1, 0])
    for inc, color in INCOME_COLORS.items():
        mask = df["Income_Group"] == inc
        if mask.any():
            ax3.scatter(df.loc[mask, "Treatment_Access_Score"],
                        df.loc[mask, "Five_Year_Survival_Pct"],
                        c=color, label=inc, s=55, alpha=0.8,
                        edgecolors="white", linewidth=0.5)
    ax3.axhline(80, color=PINK, linewidth=1.5, linestyle="--", alpha=0.7, label="80% survival")
    ax3.axvline(80, color=BLUE, linewidth=1.5, linestyle=":", alpha=0.7, label="Treatment ≥80")
    ax3.set_xlabel("Treatment Access Score")
    ax3.set_ylabel("5-Year Survival (%)")
    ax3.set_title("Treatment Access vs Survival\nby Income Group", fontweight="bold", color=PINK, fontsize=10)
    ax3.legend(fontsize=7.5, ncol=2)

    # 4. Risk Factor PAF chart
    ax4 = fig.add_subplot(gs[1, 1])
    risk_data = {
        "Age (50+)": (30.0, False), "Dense Breast": (16.0, False),
        "Family History": (12.0, False), "Obesity": (10.0, True),
        "Physical Inactivity": (9.0, True), "Early Menarche": (8.0, False),
        "Alcohol": (7.0, True), "Nulliparity": (6.0, False),
        "Late Menopause": (5.0, False), "BRCA Mutations": (5.0, False),
        "HRT": (4.0, True), "Radiation": (1.5, False),
    }
    names_r = list(risk_data.keys())
    pafs_r  = [v[0] for v in risk_data.values()]
    mods_r  = [v[1] for v in risk_data.values()]
    colors_r = [GREEN if m else PINK for m in mods_r]
    sorted_idx = np.argsort(pafs_r)
    ax4.barh([names_r[i] for i in sorted_idx], [pafs_r[i] for i in sorted_idx],
             color=[colors_r[i] for i in sorted_idx], alpha=0.82, edgecolor="white", height=0.65)
    ax4.set_xlabel("Population Attributable Fraction (%)")
    ax4.set_title("Risk Factors by PAF%", fontweight="bold", color=PINK, fontsize=10)
    mod_patch = mpatches.Patch(color=GREEN, label="Modifiable (Lifestyle)")
    unmod_patch = mpatches.Patch(color=PINK, label="Non-Modifiable")
    ax4.legend(handles=[mod_patch, unmod_patch], fontsize=8)

    # 5. Survival by Income × Stage heatmap
    ax5 = fig.add_subplot(gs[1, 2])
    stage_income_data = np.array([
        [99.5, 99.0, 93.0, 72.0, 28.0],  # High
        [97.0, 92.0, 82.0, 55.0, 18.0],  # UpperMid
        [90.0, 80.0, 65.0, 38.0, 10.0],  # LowerMid
        [82.0, 65.0, 45.0, 22.0,  5.0],  # Low
    ])
    im = ax5.imshow(stage_income_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax5, label="5-Year Survival (%)")
    ax5.set_xticks(range(5))
    ax5.set_xticklabels(["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"], fontsize=8)
    ax5.set_yticks(range(4))
    ax5.set_yticklabels(["High Income", "Upper Mid", "Lower Mid", "Low Income"], fontsize=8)
    for i in range(4):
        for j in range(5):
            ax5.text(j, i, f"{stage_income_data[i,j]:.0f}%", ha="center", va="center",
                     fontsize=8, color="white" if stage_income_data[i,j] < 50 else "black",
                     fontweight="bold")
    ax5.set_title("5-Year Survival:\nIncome × Stage Matrix", fontweight="bold", color=PINK, fontsize=10)

    fig.suptitle("🌍 WHO GBCI Policy Dashboard — 60-60-80 Target Analysis",
                 fontsize=15, fontweight="bold", color=PINK, y=1.005)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")
