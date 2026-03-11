"""
MAIN PIPELINE — Breast Cancer Global ML Project
Run: python main.py
Outputs all results, plots, and reports to reports/ and models/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import (load_data, engineer_features, create_risk_tier_labels,
                           get_feature_sets, prepare_X_y, add_pca_features)
from src.models import (get_regression_models, get_classification_models,
                        get_stacking_regressor, loocv_regression, loocv_classification,
                        cluster_countries, label_clusters_by_survival,
                        evaluate_regression, evaluate_classification)
from src.visualize import (plot_eda_overview, plot_model_comparison, plot_pred_vs_actual,
                            plot_clusters_pca, plot_who_policy_dashboard)
from src.explain import (plot_feature_importance_comparison, plot_pdp_top_features,
                          plot_survival_gap_analysis, counterfactual_analysis)

os.makedirs("reports", exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

print("="*65)
print("  🎀  BREAST CANCER GLOBAL ML PROJECT")
print("  WHO GBCI — Predictive Analytics Pipeline")
print("="*65)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD & FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Loading data & engineering features...")
df_country, df_risk, df_stage = load_data("data/raw")
df = engineer_features(df_country, df_risk, df_stage)
df = create_risk_tier_labels(df)
df.to_csv("data/processed/engineered_features.csv", index=False)
print(f"  ✓ Dataset: {df.shape[0]} countries × {df.shape[1]} features")
print(f"  ✓ Saved: data/processed/engineered_features.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: EDA VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Generating EDA visualizations...")
plot_eda_overview(df, save_path="reports/01_eda_overview.png")
plot_who_policy_dashboard(df, save_path="reports/02_who_policy_dashboard.png")
plot_survival_gap_analysis(df, list(df.columns), save_path="reports/03_survival_gap_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: CLUSTERING — RISK TIERS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Clustering countries into Risk Tiers...")
cluster_features = ["Incidence_Rate_Per_100K", "Mortality_Rate_Per_100K",
                    "Five_Year_Survival_Pct", "Stage_I_II_Pct",
                    "Treatment_Access_Score", "Mammography_Coverage_Pct"]
df, kmeans_model, cluster_scaler, silhouette = cluster_countries(df, cluster_features, n_clusters=3)
df, cluster_label_map = label_clusters_by_survival(df)
print(f"  ✓ K-Means Silhouette Score: {silhouette:.3f}")
print(f"  ✓ Cluster → Risk mapping: {cluster_label_map}")

# Save cluster viz
X_cluster = df[cluster_features].fillna(df[cluster_features].median())
X_cluster_scaled = cluster_scaler.transform(X_cluster)
plot_clusters_pca(df, X_cluster_scaled, "Cluster_KMeans", "Risk_Tier_Cluster",
                  save_path="reports/04_country_clusters_pca.png")

# Risk tier distribution
tier_counts = df["Risk_Tier_Cluster"].value_counts()
print(f"  ✓ Risk tiers: {dict(tier_counts)}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: REGRESSION — Five_Year_Survival_Pct
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Training regression models for 5-Year Survival...")
survival_features, stage_features, tier_features = get_feature_sets()
X_surv, y_surv, feat_names_surv = prepare_X_y(df, survival_features, "Five_Year_Survival_Pct")

# Scale
scaler_surv = StandardScaler()
X_surv_scaled = scaler_surv.fit_transform(X_surv)

reg_models = get_regression_models()
reg_results = []
loocv_preds_surv = {}
loo = LeaveOneOut()

for name, model in reg_models.items():
    print(f"  Training: {name}...")
    try:
        y_pred = cross_val_predict(model, X_surv_scaled, y_surv, cv=loo)
        res = evaluate_regression(y_surv, y_pred, model_name=name)
        reg_results.append(res)
        loocv_preds_surv[name] = y_pred
    except Exception as e:
        print(f"    ⚠ Skipped {name}: {e}")

# Stacking ensemble
print("  Training: Stacking Ensemble...")
try:
    stacker = get_stacking_regressor()
    y_pred_stack = cross_val_predict(stacker, X_surv_scaled, y_surv, cv=5)
    # Extend to full length via 5-fold CV
    res_stack = evaluate_regression(y_surv, y_pred_stack, model_name="Stacking Ensemble")
    reg_results.append(res_stack)
    loocv_preds_surv["Stacking Ensemble"] = y_pred_stack
except Exception as e:
    print(f"    ⚠ Stacking failed: {e}")

# Save results
reg_results_df = pd.DataFrame(reg_results)
reg_results_df.to_csv("reports/survival_regression_results.csv", index=False)
print(f"\n  ─── SURVIVAL REGRESSION LEADERBOARD ───")
print(reg_results_df[["Model", "MAE", "RMSE", "R²", "Within ±5% (%)"]].to_string(index=False))

# Plots
plot_model_comparison(reg_results, ["MAE", "R²", "Within ±5% (%)"],
                      "Model Comparison — 5-Year Survival Prediction (LOOCV)",
                      save_path="reports/05_survival_model_comparison.png")
plot_pred_vs_actual(y_surv, loocv_preds_surv, "5-Year Survival (%)",
                    save_path="reports/06_survival_pred_vs_actual.png")

# Best regression model — fit on all data
best_surv_model_name = reg_results_df.sort_values("R²", ascending=False).iloc[0]["Model"]
print(f"\n  ✓ Best survival model: {best_surv_model_name}")
best_surv_model = reg_models.get(best_surv_model_name)
if best_surv_model is None:
    best_surv_model = get_stacking_regressor()
best_surv_model.fit(X_surv_scaled, y_surv)
joblib.dump({"model": best_surv_model, "scaler": scaler_surv,
             "features": feat_names_surv, "name": best_surv_model_name},
            "models/best_survival_model.joblib")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: REGRESSION — Stage_I_II_Pct
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Training regression models for Stage I/II %...")
X_stage, y_stage, feat_names_stage = prepare_X_y(df, stage_features, "Stage_I_II_Pct")
scaler_stage = StandardScaler()
X_stage_scaled = scaler_stage.fit_transform(X_stage)

stage_results = []
loocv_preds_stage = {}

for name, model in get_regression_models().items():
    try:
        y_pred = cross_val_predict(model, X_stage_scaled, y_stage, cv=loo)
        res = evaluate_regression(y_stage, y_pred, model_name=name, verbose=False)
        stage_results.append(res)
        loocv_preds_stage[name] = y_pred
    except Exception as e:
        pass

stage_results_df = pd.DataFrame(stage_results)
stage_results_df.to_csv("reports/stage_regression_results.csv", index=False)
print(f"\n  ─── STAGE I/II REGRESSION LEADERBOARD ───")
print(stage_results_df[["Model", "MAE", "R²", "Within ±5% (%)"]].to_string(index=False))

best_stage_name = stage_results_df.sort_values("R²", ascending=False).iloc[0]["Model"]
best_stage_model = get_regression_models()[best_stage_name]
best_stage_model.fit(X_stage_scaled, y_stage)
joblib.dump({"model": best_stage_model, "scaler": scaler_stage,
             "features": feat_names_stage, "name": best_stage_name},
            "models/best_stage_model.joblib")
print(f"  ✓ Best stage model: {best_stage_name}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: CLASSIFICATION — Risk Tier
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Training classification models for Risk Tier...")
X_tier, y_tier_str, feat_names_tier = prepare_X_y(df, tier_features, "Risk_Tier")
if "Risk_Tier_Cluster" in df.columns:
    y_tier_str = df["Risk_Tier_Cluster"].values

le = LabelEncoder()
y_tier = le.fit_transform(y_tier_str)
scaler_tier = StandardScaler()
X_tier_scaled = scaler_tier.fit_transform(X_tier)

clf_models = get_classification_models()
clf_results = []

for name, model in clf_models.items():
    print(f"  Training: {name}...")
    try:
        y_pred_enc = cross_val_predict(model, X_tier_scaled, y_tier, cv=loo)
        y_pred = le.inverse_transform(y_pred_enc)
        y_true_labels = le.inverse_transform(y_tier)
        res, cm = evaluate_classification(y_true_labels, y_pred, model_name=name)
        clf_results.append(res)
    except Exception as e:
        print(f"    ⚠ {name}: {e}")

clf_results_df = pd.DataFrame(clf_results)
clf_results_df.to_csv("reports/classification_results.csv", index=False)
print(f"\n  ─── RISK TIER CLASSIFICATION LEADERBOARD ───")
print(clf_results_df[["Model", "F1 Macro", "Recall High-Risk", "Balanced Accuracy",
                       "Cohen's Kappa"]].to_string(index=False))

plot_model_comparison(clf_results, ["F1 Macro", "Balanced Accuracy", "Recall High-Risk"],
                      "Model Comparison — Risk Tier Classification (LOOCV)",
                      save_path="reports/07_classification_model_comparison.png")

# Best classifier
best_clf_name = clf_results_df.sort_values("F1 Macro", ascending=False).iloc[0]["Model"]
best_clf = clf_models[best_clf_name]
best_clf.fit(X_tier_scaled, y_tier)
joblib.dump({"model": best_clf, "scaler": scaler_tier, "le": le,
             "features": feat_names_tier, "name": best_clf_name},
            "models/best_classifier.joblib")
print(f"  ✓ Best classifier: {best_clf_name}")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7: EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Generating explainability plots...")

# Feature importance comparison (top 3 models)
top3_models = {}
for name in reg_results_df.sort_values("R²", ascending=False).head(3)["Model"]:
    m = reg_models.get(name)
    if m is not None:
        top3_models[name] = m

if top3_models:
    plot_feature_importance_comparison(
        top3_models, X_surv_scaled, y_surv, feat_names_surv,
        save_path="reports/08_feature_importance_comparison.png"
    )

# PDP for best model
try:
    best_perm = list(top3_models.values())[0]
    best_perm.fit(X_surv_scaled, y_surv)
    from sklearn.inspection import permutation_importance as perm_imp
    r = perm_imp(best_perm, X_surv_scaled, y_surv, n_repeats=20, random_state=42)
    top5_idx = np.argsort(r.importances_mean)[::-1][:6]
    plot_pdp_top_features(best_perm, X_surv_scaled, feat_names_surv,
                          list(top5_idx), "5-Year Survival (%)",
                          save_path="reports/09_partial_dependence_plots.png")
except Exception as e:
    print(f"  ⚠ PDP failed: {e}")

# Counterfactual — worst performing countries
print("\n  ── Counterfactual Analysis: Bottom 5 Countries ──")
bottom5_idx = np.argsort(y_surv)[:5]
feature_ranges = {
    "Mammography_Coverage_Pct": (0, 80),
    "Treatment_Access_Score":   (15, 96),
    "Stage_I_II_Pct":           (10, 73),
}
cf_results = []
for idx in bottom5_idx:
    country = df["Country"].iloc[idx]
    current = y_surv[idx]
    target  = min(current + 15, 85)
    cf = counterfactual_analysis(
        best_surv_model, X_surv_scaled, feat_names_surv,
        idx, country, target, current, feature_ranges
    )
    cf["Country"] = country
    cf["Current_Survival"] = current
    cf_results.append(cf)

if cf_results:
    all_cf = pd.concat(cf_results)
    all_cf.to_csv("reports/counterfactual_analysis.csv", index=False)
    print(all_cf[["Country", "Feature", "Current Value", "Required Value",
                   "Survival Gain"]].to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL REPORT: WHO Policy Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  📋  WHO GBCI POLICY SUMMARY")
print("="*65)

total = len(df)
n_early_ok = (df["Stage_I_II_Pct"] >= 60).sum()
n_treat_ok = (df["Treatment_Access_Score"] >= 80).sum()
n_both_ok  = ((df["Stage_I_II_Pct"] >= 60) & (df["Treatment_Access_Score"] >= 80)).sum()
mean_surv  = df["Five_Year_Survival_Pct"].mean()
surv_gap   = df["Five_Year_Survival_Pct"].max() - df["Five_Year_Survival_Pct"].min()

print(f"\n  🌍 Countries Analyzed:              {total}")
print(f"  📊 Mean 5-Year Survival:            {mean_surv:.1f}%")
print(f"  📉 Survival Gap (Max − Min):        {surv_gap:.1f} percentage points")
print(f"\n  WHO 60-60-80 Target Compliance:")
print(f"  ✅ Early Detection ≥60%:            {n_early_ok}/{total} ({n_early_ok/total*100:.0f}%)")
print(f"  ✅ Treatment Access ≥80:            {n_treat_ok}/{total} ({n_treat_ok/total*100:.0f}%)")
print(f"  ✅ Both Targets Met:                {n_both_ok}/{total} ({n_both_ok/total*100:.0f}%)")

print(f"\n  🔴 High-Risk Countries (top priority):")
if "Risk_Tier_Cluster" in df.columns:
    high_risk = df[df["Risk_Tier_Cluster"] == "High"][["Country", "Five_Year_Survival_Pct",
                                                         "Stage_I_II_Pct", "Treatment_Access_Score"]]
    print(high_risk.to_string(index=False))

print(f"\n  🟢 Best Practice Countries:")
if "Risk_Tier_Cluster" in df.columns:
    low_risk = df[df["Risk_Tier_Cluster"] == "Low"].nlargest(5, "Five_Year_Survival_Pct")[
        ["Country", "Five_Year_Survival_Pct", "Stage_I_II_Pct"]]
    print(low_risk.to_string(index=False))

print("\n  📁 All outputs saved to reports/ and models/")
print(f"\n  Reports generated:")
for f in sorted(os.listdir("reports")):
    size = os.path.getsize(f"reports/{f}") // 1024
    print(f"    reports/{f}  ({size} KB)")

print("\n" + "="*65)
print("  ✅  PIPELINE COMPLETE")
print("="*65)
