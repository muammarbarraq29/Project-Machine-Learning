"""
Streamlit Dashboard — Breast Cancer Global ML Project
Run: streamlit run app/streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os, sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import load_data, engineer_features, create_risk_tier_labels, prepare_X_y, get_feature_sets
from sklearn.preprocessing import StandardScaler

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎀 Breast Cancer Global ML Dashboard",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded",
)

PINK = "#C0185B"
PINK_MID = "#E91E8C"
PLIGHT = "#FCE4EC"
GREEN = "#2E7D32"
BLUE = "#1565C0"
ORANGE = "#E65100"
GRAY = "#546E7A"
TIER_COLORS = {"High": "#C0185B", "Mid": "#E65100", "Low": "#2E7D32"}
INCOME_COLORS = {"Low": "#E65100", "LowerMid": "#E91E8C", "UpperMid": "#1565C0", "High": "#2E7D32"}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    df_c, df_r, df_s = load_data("data/raw")
    df = engineer_features(df_c, df_r, df_s)
    df = create_risk_tier_labels(df)

    # Add cluster tiers if processed file exists
    try:
        df_proc = pd.read_csv("data/processed/engineered_features.csv")
        if "Risk_Tier_Cluster" in df_proc.columns:
            df["Risk_Tier_Cluster"] = df_proc["Risk_Tier_Cluster"].values
    except Exception:
        df["Risk_Tier_Cluster"] = df["Risk_Tier"].astype(str)
    return df, df_r, df_s

@st.cache_resource
def load_models():
    models = {}
    try:
        models["survival"] = joblib.load("models/best_survival_model.joblib")
    except Exception:
        pass
    try:
        models["stage"] = joblib.load("models/best_stage_model.joblib")
    except Exception:
        pass
    try:
        models["classifier"] = joblib.load("models/best_classifier.joblib")
    except Exception:
        pass
    return models

df, df_risk, df_stage = load_all()
models = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{PINK};'>🎀 BC Global ML</h2>", unsafe_allow_html=True)
    st.markdown("**WHO GBCI — Predictive Analytics**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🌍 Country Explorer",
        "🔮 Predict New Country",
        "📊 Global Dashboard",
        "🧬 Risk Factor Analysis",
        "🏆 Model Performance",
    ])
    st.markdown("---")
    st.markdown(f"<small style='color:{GRAY};'>Dataset: 50 countries · 2022–2025<br>WHO GBCI · Kaggle (zkskhurram)</small>",
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1: Country Explorer
# ─────────────────────────────────────────────────────────────────────────────
if page == "🌍 Country Explorer":
    st.markdown(f"<h1 style='color:{PINK};'>🌍 Country Explorer</h1>", unsafe_allow_html=True)
    st.markdown("Explore breast cancer statistics, risk tiers, and WHO GBCI target gaps for any country.")

    country = st.selectbox("Select Country", sorted(df["Country"].tolist()))
    row = df[df["Country"] == country].iloc[0]

    tier = str(row.get("Risk_Tier_Cluster", row.get("Risk_Tier", "N/A")))
    tier_color = TIER_COLORS.get(tier, GRAY)

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("5-Year Survival", f"{row['Five_Year_Survival_Pct']:.1f}%",
                f"{row['Five_Year_Survival_Pct'] - df['Five_Year_Survival_Pct'].mean():.1f}% vs avg")
    col2.metric("Stage I/II %", f"{row['Stage_I_II_Pct']:.0f}%",
                f"{row['Stage_I_II_Pct'] - 60:.0f}% vs WHO 60% target")
    col3.metric("Treatment Access", f"{row['Treatment_Access_Score']:.0f}/100",
                f"{row['Treatment_Access_Score'] - 80:.0f} vs WHO 80 target")
    col4.metric("Mammography Cover", f"{row['Mammography_Coverage_Pct']:.1f}%")
    col5.metric("Risk Tier", tier, delta_color="off")

    st.markdown("---")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        # WHO 60-60-80 compliance gauges
        st.markdown(f"**WHO 60-60-80 Compliance**")
        targets = {
            "Early Detection (≥60%)": (row["Stage_I_II_Pct"], 60, 100),
            "Treatment Access (≥80)": (row["Treatment_Access_Score"], 80, 100),
        }
        fig, ax = plt.subplots(figsize=(6, 3))
        for i, (label, (val, target, max_val)) in enumerate(targets.items()):
            color = GREEN if val >= target else PINK
            ax.barh(i, val, color=color, height=0.5, alpha=0.85)
            ax.barh(i, max_val - val, left=val, color="#EEEEEE", height=0.5)
            ax.axvline(target, color=BLUE, linewidth=2, linestyle="--", alpha=0.8)
            ax.text(val + 1, i, f"{val:.0f}", va="center", fontsize=11, fontweight="bold")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(list(targets.keys()), fontsize=9)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Score / %")
        ax.set_title(f"WHO Target Compliance — {country}", fontweight="bold", color=PINK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Country Profile vs Global Averages**")
        metrics = ["Five_Year_Survival_Pct", "Stage_I_II_Pct",
                   "Mammography_Coverage_Pct", "Treatment_Access_Score",
                   "Incidence_Rate_Per_100K"]
        country_vals = [row[m] for m in metrics]
        global_means = [df[m].mean() for m in metrics]
        labels = ["Survival %", "Stage I/II %", "Mammography %", "Treatment", "Incidence/100K"]

        x = np.arange(len(labels))
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        bars1 = ax2.bar(x - 0.2, country_vals, 0.38, label=country, color=PINK, alpha=0.85)
        bars2 = ax2.bar(x + 0.2, global_means, 0.38, label="Global Avg", color=BLUE, alpha=0.6)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=25, fontsize=8, ha="right")
        ax2.legend(fontsize=8)
        ax2.set_title("Country vs Global Average", fontweight="bold", color=PINK)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        st.pyplot(fig2)
        plt.close()

    # Survival gap
    if "Survival_Gap_vs_Benchmark" in df.columns:
        gap = row["Survival_Gap_vs_Benchmark"]
        inc_group = row.get("Income_Group", "N/A")
        benchmark = row.get("Expected_Survival_ByIncome", "N/A")
        color_gap = "🟢" if gap >= 0 else "🔴"
        st.info(f"{color_gap} **Survival Gap vs {inc_group} income benchmark** ({benchmark:.1f}%): **{gap:+.1f}%**  "
                f"({'Outperforming' if gap >= 0 else 'Underperforming'} peers)")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2: Predict New Country
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Predict New Country":
    st.markdown(f"<h1 style='color:{PINK};'>🔮 Predict New Country</h1>", unsafe_allow_html=True)
    st.markdown("Input health system features to predict **5-year survival**, **Stage I/II %**, and **risk tier**.")

    col1, col2 = st.columns(2)
    with col1:
        incidence   = st.slider("Incidence Rate / 100K",   10.0, 120.0, 45.0, 0.5)
        mortality   = st.slider("Mortality Rate / 100K",    4.0,  22.0,  12.0, 0.1)
        mammography = st.slider("Mammography Coverage %",   0.0,  85.0,  25.0, 0.5)
        treatment   = st.slider("Treatment Access Score",   10,   100,   50,   1)
        screening   = st.toggle("National Screening Program", value=False)
    with col2:
        new_cases   = st.number_input("New Cases 2022",  500, 400000, 15000, 500)
        population  = st.number_input("Population (M)",  1.0, 1500.0, 50.0, 1.0)
        continent   = st.selectbox("Continent",
            ["Africa", "Americas", "Asia", "Europe", "Oceania"])
        region      = st.selectbox("Region",
            sorted(df["Region"].unique().tolist()))

    if st.button("🔮 Predict", type="primary"):
        # Build a one-row DataFrame
        cont_mean = df[df["Continent"] == continent]["Five_Year_Survival_Pct"].mean() if continent in df["Continent"].values else df["Five_Year_Survival_Pct"].mean()
        reg_inc_mean = df[df["Region"] == region]["Incidence_Rate_Per_100K"].mean() if region in df["Region"].values else df["Incidence_Rate_Per_100K"].mean()
        reg_screen_pct = df[df["Region"] == region]["Screening_Program"].astype(int).mean() * 100 if region in df["Region"].values else 50.0

        def income_group(score):
            if score < 40: return "Low"
            elif score < 60: return "LowerMid"
            elif score < 80: return "UpperMid"
            else: return "High"

        ig = income_group(treatment)
        income_survival_map = {"High": 87.8, "UpperMid": 70.9, "LowerMid": 47.3, "Low": 31.1}
        expected_surv = income_survival_map[ig]

        row_pred = {
            "Incidence_Rate_Per_100K": incidence,
            "Mortality_Rate_Per_100K": mortality,
            "Mammography_Coverage_Pct": mammography,
            "Treatment_Access_Score": treatment,
            "Screening_Program": int(screening),
            "Healthcare_Quality_Index": (treatment + mammography * 0.8 + int(screening) * 20) / 3,
            "Mortality_Burden_Index": (new_cases * mortality / 100) / population * 10,
            "Case_Fatality_Ratio": mortality / incidence * 100,
            "Detection_Efficiency_Score": 40 * (mammography / 100),  # placeholder
            "Income_Group": {"Low": 0, "LowerMid": 1, "UpperMid": 2, "High": 3}[ig],
            "Mammography_x_Treatment": mammography * treatment,
            "Screening_x_Stage": int(screening) * 40,  # placeholder
            "Log_New_Cases": np.log1p(new_cases),
            "Log_Population": np.log1p(population),
            "Mammography_Squared": mammography ** 2,
            "Continent_Survival_Encoded": cont_mean,
            "Region_Incidence_Mean": reg_inc_mean,
            "Region_Screening_Penetration": reg_screen_pct,
            "WHO_EarlyDetection_Gap": 60 - 40,  # placeholder
            "WHO_Treatment_Gap": 80 - treatment,
            "StageShift_Potential": 40,  # placeholder
        }

        X_new = np.array([[row_pred[f] for f in models["survival"]["features"]]])
        scaler_s = models["survival"]["scaler"]
        X_new_s  = scaler_s.transform(X_new)
        surv_pred = models["survival"]["model"].predict(X_new_s)[0]
        surv_pred = np.clip(surv_pred, 20, 98)

        X_new2 = np.array([[row_pred[f] for f in models["stage"]["features"]]])
        scaler_st = models["stage"]["scaler"]
        X_new2_s  = scaler_st.transform(X_new2)
        stage_pred = models["stage"]["model"].predict(X_new2_s)[0]
        stage_pred = np.clip(stage_pred, 8, 75)

        # Risk tier via heuristic
        score = ((mortality / 22) * 35 + ((100 - surv_pred) / 100) * 35 + ((100 - stage_pred) / 100) * 30)
        tier_pred = "High" if score > 45 else ("Mid" if score > 28 else "Low")
        tier_color = TIER_COLORS[tier_pred]

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("🎯 Predicted 5-Year Survival", f"{surv_pred:.1f}%",
                        f"{surv_pred - df['Five_Year_Survival_Pct'].mean():.1f}% vs global avg")
        res_col2.metric("📋 Predicted Stage I/II %", f"{stage_pred:.1f}%",
                        f"{stage_pred - 60:.1f}% vs WHO 60% target")
        res_col3.metric("⚠ Predicted Risk Tier", tier_pred, delta_color="off")

        # WHO compliance
        st.markdown(f"""
        **WHO 60-60-80 Assessment:**
        - Early Detection: {'✅' if stage_pred >= 60 else '❌'} {stage_pred:.0f}% (target: 60%)
        - Treatment Access: {'✅' if treatment >= 80 else '❌'} {treatment}/100 (target: 80)
        - Survival Gap vs {ig} benchmark ({expected_surv:.0f}%): **{surv_pred - expected_surv:+.1f}%**

        **Priority Action:** {'Urgent — screening programme + treatment capacity investment needed' if tier_pred == 'High' else ('Scale screening coverage to reach 60% target' if tier_pred == 'Mid' else 'Sustain programmes and share best practices')}
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3: Global Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Global Dashboard":
    st.markdown(f"<h1 style='color:{PINK};'>📊 Global Dashboard</h1>", unsafe_allow_html=True)

    # Show pregenerated images if available
    report_imgs = {
        "EDA Overview": "reports/01_eda_overview.png",
        "WHO Policy Dashboard": "reports/02_who_policy_dashboard.png",
        "Survival Gap Analysis": "reports/03_survival_gap_analysis.png",
        "Country Clusters (PCA)": "reports/04_country_clusters_pca.png",
    }

    for title, path in report_imgs.items():
        if os.path.exists(path):
            st.markdown(f"### {title}")
            st.image(path, use_container_width=True)
            st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4: Risk Factor Analysis
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧬 Risk Factor Analysis":
    st.markdown(f"<h1 style='color:{PINK};'>🧬 Risk Factor Analysis</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            df_risk[["Risk_Factor", "Relative_Risk", "Population_Attributable_Fraction_Pct",
                     "Evidence_Level", "Modifiable"]].sort_values(
                "Population_Attributable_Fraction_Pct", ascending=False
            ).style.background_gradient(cmap="RdYlGn", subset=["Relative_Risk"]),
            use_container_width=True
        )

    with col2:
        fig, ax = plt.subplots(figsize=(6, 5))
        df_risk_s = df_risk.sort_values("Population_Attributable_Fraction_Pct")
        colors = [GREEN if m else PINK for m in df_risk_s["Modifiable"]]
        ax.barh(df_risk_s["Risk_Factor"], df_risk_s["Population_Attributable_Fraction_Pct"],
                color=colors, alpha=0.82, edgecolor="white")
        ax.set_xlabel("Population Attributable Fraction (%)")
        ax.set_title("Risk Factors by PAF%", fontweight="bold", color=PINK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        mod_patch  = mpatches.Patch(color=GREEN, label="Modifiable")
        unmod_patch = mpatches.Patch(color=PINK, label="Non-Modifiable")
        ax.legend(handles=[mod_patch, unmod_patch])
        st.pyplot(fig)
        plt.close()

    st.markdown("### Modifiable vs Non-Modifiable Risk Burden")
    mod_total   = df_risk[df_risk["Modifiable"] == True]["Population_Attributable_Fraction_Pct"].sum()
    unmod_total = df_risk[df_risk["Modifiable"] == False]["Population_Attributable_Fraction_Pct"].sum()
    st.info(f"🟢 **Modifiable lifestyle factors** account for **{mod_total:.0f}%** total PAF — "
            f"representing the intervention opportunity for public health programmes.")
    st.info(f"🔴 **Non-modifiable factors** account for **{unmod_total:.0f}%** PAF — "
            f"informing genetic screening and high-risk surveillance protocols.")

    # Stage × Income heatmap
    st.markdown("### 5-Year Survival: Stage × Income Matrix")
    stage_data = {
        "Stage": ["Stage 0", "Stage I", "Stage II", "Stage III", "Stage IV"],
        "High Income": [99.5, 99.0, 93.0, 72.0, 28.0],
        "Upper Middle": [97.0, 92.0, 82.0, 55.0, 18.0],
        "Lower Middle": [90.0, 80.0, 65.0, 38.0, 10.0],
        "Low Income": [82.0, 65.0, 45.0, 22.0, 5.0],
    }
    st.dataframe(
        pd.DataFrame(stage_data).set_index("Stage").style.background_gradient(
            cmap="RdYlGn", vmin=0, vmax=100),
        use_container_width=True
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 5: Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏆 Model Performance":
    st.markdown(f"<h1 style='color:{PINK};'>🏆 Model Performance</h1>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Survival Regression", "Stage Regression", "Risk Classification"])

    with tab1:
        try:
            df_surv = pd.read_csv("reports/survival_regression_results.csv")
            st.dataframe(df_surv.sort_values("R²", ascending=False).style.background_gradient(
                cmap="RdYlGn", subset=["R²"]).background_gradient(
                cmap="RdYlGn_r", subset=["MAE", "RMSE"]), use_container_width=True)
        except Exception:
            st.warning("Run main.py first to generate results.")
        if os.path.exists("reports/05_survival_model_comparison.png"):
            st.image("reports/05_survival_model_comparison.png", use_container_width=True)
        if os.path.exists("reports/06_survival_pred_vs_actual.png"):
            st.image("reports/06_survival_pred_vs_actual.png", use_container_width=True)

    with tab2:
        try:
            df_stage_r = pd.read_csv("reports/stage_regression_results.csv")
            st.dataframe(df_stage_r.sort_values("R²", ascending=False).style.background_gradient(
                cmap="RdYlGn", subset=["R²"]), use_container_width=True)
        except Exception:
            st.warning("Run main.py first to generate results.")

    with tab3:
        try:
            df_clf = pd.read_csv("reports/classification_results.csv")
            st.dataframe(df_clf.sort_values("F1 Macro", ascending=False).style.background_gradient(
                cmap="RdYlGn", subset=["F1 Macro", "Balanced Accuracy", "Recall High-Risk"]),
                use_container_width=True)
        except Exception:
            st.warning("Run main.py first to generate results.")
        if os.path.exists("reports/07_classification_model_comparison.png"):
            st.image("reports/07_classification_model_comparison.png", use_container_width=True)

    st.markdown("### Feature Importance & Explainability")
    for path, title in [
        ("reports/08_feature_importance_comparison.png", "Feature Importance Comparison"),
        ("reports/09_partial_dependence_plots.png", "Partial Dependence Plots"),
        ("reports/03_survival_gap_analysis.png", "Survival Gap Analysis"),
    ]:
        if os.path.exists(path):
            st.markdown(f"**{title}**")
            st.image(path, use_container_width=True)

    try:
        df_cf = pd.read_csv("reports/counterfactual_analysis.csv")
        st.markdown("### Counterfactual Analysis — Worst 5 Countries")
        st.dataframe(df_cf.style.background_gradient(cmap="RdYlGn", subset=["Survival Gain"]),
                     use_container_width=True)
        st.caption("Shows what change in each feature would be needed to gain +15% survival in the 5 lowest-performing countries.")
    except Exception:
        pass
