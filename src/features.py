"""
Feature Engineering Module
Breast Cancer Global ML Project
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data(base_path="data/raw"):
    df_country = pd.read_csv(f"{base_path}/breast_cancer_by_country.csv")
    df_risk = pd.read_csv(f"{base_path}/breast_cancer_risk_factors.csv")
    df_stage = pd.read_csv(f"{base_path}/breast_cancer_survival_by_stage.csv")
    return df_country, df_risk, df_stage


def engineer_features(df, df_risk, df_stage):
    df = df.copy()

    # ── Tier 1: Composite Health System Indices ──────────────────────────
    df["Healthcare_Quality_Index"] = (
        df["Treatment_Access_Score"]
        + df["Mammography_Coverage_Pct"] * 0.8
        + df["Screening_Program"].astype(int) * 20
    ) / 3

    df["Survival_Efficiency_Ratio"] = (
        df["Five_Year_Survival_Pct"] / df["Incidence_Rate_Per_100K"]
    )

    df["Detection_Efficiency_Score"] = (
        df["Stage_I_II_Pct"] * (df["Mammography_Coverage_Pct"] / 100)
    )

    df["Mortality_Burden_Index"] = df["Deaths_2022"] / df["Population_M"] * 10

    df["Case_Fatality_Ratio"] = df["Deaths_2022"] / df["New_Cases_2022"] * 100

    # ── Tier 2: Income group derivation ─────────────────────────────────
    def assign_income(score):
        if score < 40:   return "Low"
        elif score < 60: return "LowerMid"
        elif score < 80: return "UpperMid"
        else:            return "High"

    df["Income_Group"] = df["Treatment_Access_Score"].apply(assign_income)

    # Expected benchmark survival from Table 3 by income group
    income_survival_map = {
        "High":      (0.20*99.5 + 0.30*99.0 + 0.35*93.0 + 0.12*72.0 + 0.03*28.0),
        "UpperMid":  (0.12*97.0 + 0.18*92.0 + 0.25*82.0 + 0.22*55.0 + 0.23*18.0),
        "LowerMid":  (0.05*90.0 + 0.10*80.0 + 0.18*65.0 + 0.30*38.0 + 0.37*10.0),
        "Low":       (0.02*82.0 + 0.05*65.0 + 0.12*45.0 + 0.35*22.0 + 0.46*5.0),
    }
    df["Expected_Survival_ByIncome"] = df["Income_Group"].map(income_survival_map)
    df["Survival_Gap_vs_Benchmark"] = (
        df["Five_Year_Survival_Pct"] - df["Expected_Survival_ByIncome"]
    )

    # Max Stage_I_II_Pct for each income tier as ceiling
    income_stage_ceiling = {"High": 66, "UpperMid": 55, "LowerMid": 35, "Low": 17}
    df["StageShift_Potential"] = df["Income_Group"].map(income_stage_ceiling) - df["Stage_I_II_Pct"]

    # ── Tier 3: Risk Factor Aggregation ─────────────────────────────────
    mod_paf_sum = df_risk[df_risk["Modifiable"] == True]["Population_Attributable_Fraction_Pct"].sum()
    unmod_paf_sum = df_risk[df_risk["Modifiable"] == False]["Population_Attributable_Fraction_Pct"].sum()
    max_rr_mod = df_risk[df_risk["Modifiable"] == True]["Relative_Risk"].max()
    weighted_paf = (df_risk["Population_Attributable_Fraction_Pct"] * df_risk["Relative_Risk"]).sum()

    # These are global constants — assign to all countries (enrich with regional mapping later)
    df["Modifiable_Risk_PAF_Sum"] = mod_paf_sum
    df["Lifestyle_vs_Genetic_Ratio"] = mod_paf_sum / unmod_paf_sum
    df["Max_RR_Modifiable"] = max_rr_mod
    df["Weighted_PAF_Score"] = weighted_paf

    # ── Tier 4: Interaction Terms ────────────────────────────────────────
    df["Mammography_x_Treatment"] = (
        df["Mammography_Coverage_Pct"] * df["Treatment_Access_Score"]
    )
    df["Screening_x_Stage"] = (
        df["Screening_Program"].astype(int) * df["Stage_I_II_Pct"]
    )
    df["Incidence_x_HQI"] = df["Incidence_Rate_Per_100K"] * df["Healthcare_Quality_Index"]
    df["Log_New_Cases"] = np.log1p(df["New_Cases_2022"])
    df["Log_Population"] = np.log1p(df["Population_M"])
    df["Mammography_Squared"] = df["Mammography_Coverage_Pct"] ** 2

    # ── Tier 5: Regional Encodings ───────────────────────────────────────
    continent_survival_mean = df.groupby("Continent")["Five_Year_Survival_Pct"].transform("mean")
    df["Continent_Survival_Encoded"] = continent_survival_mean

    region_incidence_mean = df.groupby("Region")["Incidence_Rate_Per_100K"].transform("mean")
    df["Region_Incidence_Mean"] = region_incidence_mean

    region_screening_pct = df.groupby("Region")["Screening_Program"].transform(
        lambda x: x.astype(int).mean() * 100
    )
    df["Region_Screening_Penetration"] = region_screening_pct

    # WHO 60-60-80 compliance flags
    df["WHO_EarlyDetection_Gap"] = 60 - df["Stage_I_II_Pct"]
    df["WHO_Treatment_Gap"] = 80 - df["Treatment_Access_Score"]
    df["WHO_Fully_Compliant"] = (
        (df["Stage_I_II_Pct"] >= 60) & (df["Treatment_Access_Score"] >= 80)
    ).astype(int)

    return df


def create_risk_tier_labels(df):
    """Create 3-class risk tier based on composite score"""
    score = (
        (df["Mortality_Rate_Per_100K"] / df["Mortality_Rate_Per_100K"].max()) * 35
        + ((100 - df["Five_Year_Survival_Pct"]) / 100) * 35
        + ((100 - df["Stage_I_II_Pct"]) / 100) * 30
    )
    df["Risk_Score_Raw"] = score
    terciles = score.quantile([1/3, 2/3])
    df["Risk_Tier"] = pd.cut(
        score,
        bins=[-np.inf, terciles.iloc[0], terciles.iloc[1], np.inf],
        labels=["Low", "Mid", "High"]
    )
    return df


def get_feature_sets():
    """Return feature column lists for each modelling task"""
    base_features = [
        "Incidence_Rate_Per_100K", "Mortality_Rate_Per_100K",
        "Mammography_Coverage_Pct", "Treatment_Access_Score",
        "Screening_Program",
        "Healthcare_Quality_Index", "Mortality_Burden_Index",
        "Case_Fatality_Ratio", "Detection_Efficiency_Score",
        "Income_Group",
        "Mammography_x_Treatment", "Screening_x_Stage",
        "Log_New_Cases", "Log_Population", "Mammography_Squared",
        "Continent_Survival_Encoded", "Region_Incidence_Mean",
        "Region_Screening_Penetration",
        "WHO_EarlyDetection_Gap", "WHO_Treatment_Gap",
        "StageShift_Potential",
    ]

    survival_features = base_features  # for predicting Five_Year_Survival_Pct
    stage_features = [f for f in base_features if f != "WHO_EarlyDetection_Gap"]
    tier_features = base_features

    return survival_features, stage_features, tier_features


def prepare_X_y(df, features, target):
    """Encode categoricals, return X, y arrays"""
    X = df[features].copy()

    # Encode categorical
    if "Income_Group" in X.columns:
        income_map = {"Low": 0, "LowerMid": 1, "UpperMid": 2, "High": 3}
        X["Income_Group"] = X["Income_Group"].map(income_map)

    if "Screening_Program" in X.columns:
        X["Screening_Program"] = X["Screening_Program"].astype(int)

    X = X.fillna(X.median(numeric_only=True))
    y = df[target].values
    return X.values, y, X.columns.tolist()


def add_pca_features(X, n_components=3, scaler=None, pca=None, fit=True):
    """Add PCA meta-features to X"""
    if fit:
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)
        X_scaled = scaler.fit_transform(X)
        pca_components = pca.fit_transform(X_scaled)
    else:
        X_scaled = scaler.transform(X)
        pca_components = pca.transform(X_scaled)

    X_augmented = np.hstack([X, pca_components])
    return X_augmented, scaler, pca
