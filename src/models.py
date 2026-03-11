"""
Models & Evaluation Module
Breast Cancer Global ML Project
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, balanced_accuracy_score, confusion_matrix,
    classification_report, cohen_kappa_score
)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESSION MODELS
# ══════════════════════════════════════════════════════════════════════════════

def get_regression_models():
    return {
        "Ridge Regression":          Ridge(alpha=1.0),
        "Lasso Regression":          Lasso(alpha=0.1, max_iter=5000),
        "Decision Tree":             DecisionTreeRegressor(max_depth=4, min_samples_leaf=4, random_state=42),
        "Random Forest":             RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                                           max_features="sqrt", random_state=42, oob_score=True),
        "Gradient Boosting":         GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                                learning_rate=0.05, subsample=0.8,
                                                                min_samples_leaf=3, random_state=42),
        "Extra Trees":               ExtraTreesRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                                         max_features="sqrt", random_state=42),
        "SVR (RBF)":                 SVR(kernel="rbf", C=10, epsilon=2, gamma="scale"),
        "KNN Regressor":             KNeighborsRegressor(n_neighbors=5, weights="distance"),
    }


def get_stacking_regressor():
    estimators = [
        ("rf",  RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)),
        ("gb",  GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)),
        ("et",  ExtraTreesRegressor(n_estimators=200, max_depth=5, random_state=42)),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        passthrough=False
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION MODELS
# ══════════════════════════════════════════════════════════════════════════════

def get_classification_models():
    return {
        "Logistic Regression":       LogisticRegression(C=0.1, max_iter=2000, random_state=42),
        "Decision Tree":             DecisionTreeClassifier(max_depth=4, min_samples_leaf=4, random_state=42),
        "Random Forest":             RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                                            max_features="sqrt", random_state=42, oob_score=True,
                                                            class_weight="balanced"),
        "Gradient Boosting":         GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                                  learning_rate=0.05, subsample=0.8,
                                                                  min_samples_leaf=3, random_state=42),
        "Extra Trees":               ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                                          class_weight="balanced", random_state=42),
        "SVM (RBF)":                 SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                                         class_weight="balanced", random_state=42),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_regression(y_true, y_pred, model_name="", verbose=True):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    max_err = np.max(np.abs(y_true - y_pred))
    within5 = np.mean(np.abs(y_true - y_pred) <= 5) * 100  # policy threshold
    corr, pval = pearsonr(y_true, y_pred)

    results = {
        "Model": model_name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R²": round(r2, 3),
        "MAPE (%)": round(mape, 2),
        "Max Error": round(max_err, 2),
        "Within ±5% (%)": round(within5, 1),
        "Pearson r": round(corr, 3),
        "p-value": round(pval, 4),
    }
    if verbose:
        print(f"\n{'─'*55}")
        print(f"  {model_name}")
        print(f"{'─'*55}")
        for k, v in results.items():
            if k != "Model":
                print(f"  {k:<22} {v}")
    return results


def loocv_regression(model, X, y, model_name=""):
    """Leave-One-Out Cross Validation for regression"""
    loo = LeaveOneOut()
    y_pred = cross_val_predict(model, X, y, cv=loo)
    return evaluate_regression(y, y_pred, model_name)


def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=500, ci=0.95):
    """Bootstrap confidence interval for a metric"""
    n = len(y_true)
    boot_scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        score = metric_fn(y_true[idx], y_pred[idx])
        boot_scores.append(score)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_scores, alpha * 100)
    upper = np.percentile(boot_scores, (1 - alpha) * 100)
    return np.mean(boot_scores), lower, upper


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION — CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_classification(y_true, y_pred, y_prob=None, model_name="", labels=None):
    labels = labels or ["Low", "Mid", "High"]
    f1_macro = f1_score(y_true, y_pred, average="macro", labels=labels)
    f1_high  = f1_score(y_true, y_pred, labels=["High"], average="micro")
    recall_high = np.sum((y_pred == "High") & (y_true == "High")) / max(np.sum(y_true == "High"), 1)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    kappa    = cohen_kappa_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred, labels=labels)

    results = {
        "Model": model_name,
        "F1 Macro": round(f1_macro, 3),
        "F1 High-Risk": round(f1_high, 3),
        "Recall High-Risk": round(recall_high, 3),
        "Balanced Accuracy": round(bal_acc, 3),
        "Cohen's Kappa": round(kappa, 3),
    }
    print(f"\n{'─'*55}")
    print(f"  {model_name}")
    print(f"{'─'*55}")
    for k, v in results.items():
        if k != "Model":
            print(f"  {k:<25} {v}")
    print(f"\n  Confusion Matrix (Low/Mid/High):")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df.to_string(index=True))
    return results, cm


def loocv_classification(model, X, y_encoded, le, model_name=""):
    loo = LeaveOneOut()
    y_pred_enc = cross_val_predict(model, X, y_encoded, cv=loo)
    y_pred = le.inverse_transform(y_pred_enc)
    y_true = le.inverse_transform(y_encoded)
    return evaluate_classification(y_true, y_pred, model_name=model_name)


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING — RISK TIERS
# ══════════════════════════════════════════════════════════════════════════════

def cluster_countries(df, features, n_clusters=3):
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

    hier = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    df["Cluster_Hierarchical"] = hier.fit_predict(X_scaled)

    # Silhouette
    from sklearn.metrics import silhouette_score
    sil = silhouette_score(X_scaled, df["Cluster_KMeans"])

    return df, kmeans, scaler, sil


def label_clusters_by_survival(df):
    """Assign Low/Mid/High risk label to KMeans clusters based on mean survival"""
    cluster_survival = df.groupby("Cluster_KMeans")["Five_Year_Survival_Pct"].mean().sort_values(ascending=False)
    label_map = {
        cluster_survival.index[0]: "Low",
        cluster_survival.index[1]: "Mid",
        cluster_survival.index[2]: "High",
    }
    df["Risk_Tier_Cluster"] = df["Cluster_KMeans"].map(label_map)
    return df, label_map
