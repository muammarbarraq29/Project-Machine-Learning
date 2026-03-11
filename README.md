# 🎀 Breast Cancer Global ML Project
**WHO GBCI — Predictive Analytics | Dataset: Kaggle (zkskhurram) 2022–2025**

## Project Structure

```
bc_project/
├── data/
│   ├── raw/                    # Original CSVs (3 files)
│   └── processed/              # Engineered features (40 columns)
├── src/
│   ├── features.py             # Feature engineering (25+ features)
│   ├── models.py               # 8 models + stacking ensemble
│   ├── explain.py              # Feature importance, PDP, counterfactuals
│   └── visualize.py            # EDA + policy dashboard plots
├── app/
│   └── streamlit_app.py        # 5-tab interactive dashboard
├── models/                     # Saved .joblib models
├── reports/                    # All generated plots + CSV results
├── main.py                     # Full pipeline runner
└── requirements.txt
```

## Models Trained (LOOCV Evaluation)

**Regression — 5-Year Survival %**
| Model              | MAE   | R²    | Within ±5% |
|--------------------|-------|-------|------------|
| Ridge Regression   | 2.16% | 0.979 | 88%        |
| Lasso Regression   | 2.28% | 0.976 | 90%        |
| Gradient Boosting  | 2.46% | 0.965 | 86%        |
| Stacking Ensemble  | 2.91% | 0.958 | 84%        |

**Classification — Risk Tier (Low/Mid/High)**
| Model              | F1 Macro | Recall High-Risk | Kappa |
|--------------------|----------|-----------------|-------|
| Gradient Boosting  | 0.953    | 0.917           | 0.938 |
| SVM (RBF)         | 0.930    | 0.917           | 0.908 |
| Random Forest      | 0.936    | 0.917           | 0.908 |

## WHO GBCI Key Findings

- **Only 34%** of 50 countries meet the WHO 60% early detection target
- **68.6 percentage point** survival gap between best (South Korea 93.6%) and worst (Tanzania 25%)
- **Top predictor**: Mammography coverage + Treatment access synergy
- **12 High-Risk countries** could reach Mid-Risk with +20pt Treatment Access improvement

## Reports Generated

- `reports/01_eda_overview.png` — 10-panel EDA dashboard
- `reports/02_who_policy_dashboard.png` — WHO 60-60-80 compliance analysis
- `reports/03_survival_gap_analysis.png` — Income group gap decomposition
- `reports/04_country_clusters_pca.png` — PCA cluster visualization
- `reports/05_survival_model_comparison.png` — Model leaderboard charts
- `reports/06_survival_pred_vs_actual.png` — Predicted vs Actual (LOOCV)
- `reports/07_classification_model_comparison.png` — Classifier metrics
- `reports/08_feature_importance_comparison.png` — Permutation importance
- `reports/09_partial_dependence_plots.png` — PDP top-6 features
- `reports/counterfactual_analysis.csv` — Intervention simulations

## Streamlit Dashboard (5 Tabs)

1. **🌍 Country Explorer** — Select any country, view all metrics + WHO compliance
2. **🔮 Predict New Country** — Real-time prediction with custom feature inputs
3. **📊 Global Dashboard** — All generated visualizations
4. **🧬 Risk Factor Analysis** — PAF bubble chart, Stage × Income heatmap
5. **🏆 Model Performance** — Full results tables + explainability plots
