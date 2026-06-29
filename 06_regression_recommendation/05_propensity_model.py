#!/usr/bin/env python
"""
Piece 05 — Recommendation Propensity Model
Qatar Airways PDD Portfolio | Edwin Daniels

Predicts whether a passenger will recommend Qatar Airways (Yes/No) as a
loyalty/churn proxy. Target variable: `recommended` from real Skytrax reviews.

The primary output is not the model itself but the product investment
prioritisation framework: which experience dimensions most strongly drive
recommendation intent, with plain-English implications for PDD decision-making.

Models:
  1. Logistic Regression  — baseline + interpretable coefficients
  2. Random Forest        — non-linear importance + PDP stability
  3. XGBoost              — performance benchmark + SHAP decomposition

Usage:
  python 05_regression/05_propensity_model.py
"""

import os
import pickle
import sys
import warnings

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")                       # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Brand palette ─────────────────────────────────────────────────────────────
C = {
    "maroon":     "#8D1B3D",
    "steel_dark": "#4A6278",
    "steel_mid":  "#6B8296",
    "grey_dark":  "#5A5A5A",
    "grey_light": "#C8C9CA",
    "off_white":  "#F7F5F2",
    "positive":   "#2E7D5A",
    "negative":   "#C0392B",
}

plt.rcParams.update({
    "font.family":   "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.facecolor":  C["off_white"],
    "figure.facecolor": C["off_white"],
})

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(SCRIPT_DIR)

DATA_PATH  = os.path.join(ROOT, "06_outputs", "nlp_output_files", "qatar_sentiment_flat.csv")
CHART_DIR  = os.path.join(ROOT, "06_outputs", "charts", "05_regression_charts")
MODEL_DIR  = os.path.join(ROOT, "06_outputs", "propensity_model")
NLP_OUT    = os.path.join(ROOT, "06_outputs", "nlp_output_files")

for d in [CHART_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

DPI = 180


def save_fig(fname: str) -> None:
    path = os.path.join(CHART_DIR, fname)
    plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["off_white"])
    plt.close("all")
    print(f"    ✓ {fname}")


def clean_name(col: str) -> str:
    """Map internal feature column names to readable display labels."""
    label_map = {
        "feat_overall_rating":       "Overall Rating (1–10)",
        "feat_seat_comfort":         "Seat Comfort Score",
        "feat_cabin_staff":          "Cabin Staff Score",
        "feat_food_beverage":        "Food & Beverage Score",
        "feat_ife":                  "IFE Score",
        "feat_ground_service":       "Ground Service Score",
        "feat_wifi":                 "Wi-Fi Score",
        "feat_value_for_money":      "Value for Money Score",
        "feat_sentiment_score":      "NLP Sentiment Score",
        "feat_trip_verified":        "Trip Verified",
        "feat_is_sarcastic":         "Sarcastic Tone (NLP)",
        "feat_competitor_mentioned": "Competitor Mentioned",
        "feat_rebook":               "Would Rebook Signal",
        "feat_theme_seat_comfort":   "NLP Theme: Seat Comfort",
        "feat_theme_cabin_crew":     "NLP Theme: Cabin Crew",
        "feat_theme_food_beverage":  "NLP Theme: Food & Bev",
        "feat_theme_ife":            "NLP Theme: IFE",
        "feat_theme_wifi":           "NLP Theme: Wi-Fi",
        "feat_theme_value":          "NLP Theme: Value",
        "feat_year":                 "Year",
        "feat_month":                "Month",
    }
    if col in label_map:
        return label_map[col]
    col = col.replace("seat_", "Cabin: ").replace("traveller_", "Traveller: ").replace("emotion_", "Emotion: ")
    return col.replace("_", " ").title()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  PIECE 05 — RECOMMENDATION PROPENSITY MODEL")
print("=" * 62)
print("\n[1/5] Loading & engineering features...")

# The NLP sentiment flat file is the merged source: all 1,957 Skytrax reviews
# enriched with LLM-extracted fields. review_id = row index of source CSV.
df = pd.read_csv(DATA_PATH)
print(f"  Rows loaded: {len(df):,}")

# ── Target: recommended → binary ─────────────────────────────────────────────
df = df[df["recommended"].isin(["Yes", "No"])].copy()
df["target"] = (df["recommended"] == "Yes").astype(int)
n_yes = df["target"].sum()
n_no  = (df["target"] == 0).sum()
print(f"  Target — Yes (recommend): {n_yes:,} ({n_yes/len(df)*100:.1f}%) | "
      f"No: {n_no:,} ({n_no/len(df)*100:.1f}%)")
print(f"  Imbalance ratio {n_yes/n_no:.2f}:1 -> using class_weight='balanced'")

# ── Score columns: "Not Rated" → NaN → float ─────────────────────────────────
score_map = {
    "score_seat_comfort":    "feat_seat_comfort",
    "score_cabin_staff":     "feat_cabin_staff",
    "score_food_beverages":  "feat_food_beverage",
    "score_ife":             "feat_ife",
    "score_ground_service":  "feat_ground_service",
    "score_wifi":            "feat_wifi",
    "score_value_for_money": "feat_value_for_money",
}
for raw, new in score_map.items():
    df[new] = pd.to_numeric(df[raw].replace("Not Rated", np.nan), errors="coerce")

# ── Overall rating ────────────────────────────────────────────────────────────
df["feat_overall_rating"] = pd.to_numeric(df["overall_rating_10"], errors="coerce")

# ── NLP numeric ──────────────────────────────────────────────────────────────
df["feat_sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")

# ── Binary flags ─────────────────────────────────────────────────────────────
_bool = {True: 1, False: 0, "True": 1, "False": 0, "Yes": 1, "No": 0, 1: 1, 0: 0}
df["feat_trip_verified"]        = df["trip_verified"].map(_bool)
df["feat_is_sarcastic"]         = df["is_sarcastic"].map(_bool)
df["feat_competitor_mentioned"] = df["competitor_mentioned"].map(_bool)

# ── Would rebook: ordinal yes=2 | maybe=1 | no=0 | not_stated=NaN ─────────────
rebook_map = {"yes": 2, "maybe": 1, "no": 0, "not_stated": np.nan}
df["feat_rebook"] = (
    df["would_rebook_signal"].astype(str).str.lower().str.strip().map(rebook_map)
)

# ── Theme columns: positive=1 | neutral=0 | negative=-1 | rest=NaN ───────────
theme_enc = {"positive": 1, "neutral": 0, "negative": -1,
             "not_mentioned": np.nan, "mixed": np.nan}
for col in ["theme_seat_comfort", "theme_cabin_crew", "theme_food_beverage",
            "theme_ife", "theme_wifi", "theme_value"]:
    df[f"feat_{col}"] = (
        df[col].astype(str).str.lower().str.strip().map(theme_enc)
    )

# ── Temporal ──────────────────────────────────────────────────────────────────
df["feat_year"]  = pd.to_numeric(df["year"], errors="coerce")
df["feat_month"] = pd.to_numeric(df["month"], errors="coerce")

# ── One-hot encode categoricals — fill NaN before encoding ───────────────────
seat_dummies      = pd.get_dummies(df["seat_type"].fillna("Unknown"),         prefix="seat")
traveller_dummies = pd.get_dummies(df["type_of_traveller"].fillna("Unknown"), prefix="traveller")
emotion_dummies   = pd.get_dummies(df["emotion_primary"].fillna("Unknown"),   prefix="emotion")

# ── Assemble feature matrix ───────────────────────────────────────────────────
feat_cols = [c for c in df.columns if c.startswith("feat_")]
X_raw = pd.concat([df[feat_cols], seat_dummies, traveller_dummies, emotion_dummies], axis=1)
y     = df["target"].values

print(f"  Feature matrix: {X_raw.shape[0]:,} rows × {X_raw.shape[1]} columns")
print(f"  Missing values pre-imputation: {X_raw.isna().sum().sum():,}")

# ── Impute with median (all numeric after encoding; dummies are complete) ─────
imputer = SimpleImputer(strategy="median")
X_arr = imputer.fit_transform(X_raw)
X_df  = pd.DataFrame(X_arr, columns=X_raw.columns)

# ── Train / test split — stratified 80/20 ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── Scaled copy for logistic regression ──────────────────────────────────────
scaler     = StandardScaler()
X_train_s  = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_s   = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Training models (5-fold CV on train set)...")

# ── Model 1: Logistic Regression ─────────────────────────────────────────────
print("  → Logistic Regression...")
lr = LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0,
                         solver="lbfgs", random_state=42)
lr_cv  = cross_val_score(lr, X_train_s, y_train, cv=cv, scoring="roc_auc")
lr.fit(X_train_s, y_train)
lr_pred  = lr.predict(X_test_s)
lr_proba = lr.predict_proba(X_test_s)[:, 1]
lr_auc   = roc_auc_score(y_test, lr_proba)
print(f"     CV AUC: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}  |  Test AUC: {lr_auc:.4f}")

# ── Model 2: Random Forest ────────────────────────────────────────────────────
print("  → Random Forest...")
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                             min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_cv  = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc")
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_auc   = roc_auc_score(y_test, rf_proba)
print(f"     CV AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}  |  Test AUC: {rf_auc:.4f}")

# ── Model 3: XGBoost ──────────────────────────────────────────────────────────
print("  → XGBoost...")
spw = (y_train == 0).sum() / (y_train == 1).sum()   # scale_pos_weight for imbalance
xgb_m = xgb.XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=spw, random_state=42,
    verbosity=0, device="cpu",
)
xgb_cv = cross_val_score(xgb_m, X_train, y_train, cv=cv, scoring="roc_auc")
xgb_m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_pred  = xgb_m.predict(X_test)
xgb_proba = xgb_m.predict_proba(X_test)[:, 1]
xgb_auc   = roc_auc_score(y_test, xgb_proba)
print(f"     CV AUC: {xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}  |  Test AUC: {xgb_auc:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EVALUATION CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Evaluation charts...")

# ── Fig 01: Confusion matrices ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Confusion Matrices — Recommendation Propensity Models",
             fontsize=14, fontweight="bold", color=C["grey_dark"])

for ax, (name, pred, auc) in zip(axes, [
    ("Logistic Regression", lr_pred,  lr_auc),
    ("Random Forest",        rf_pred,  rf_auc),
    ("XGBoost",              xgb_pred, xgb_auc),
]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                cmap=sns.light_palette(C["maroon"], as_cmap=True),
                xticklabels=["Pred: No", "Pred: Yes"],
                yticklabels=["Actual: No", "Actual: Yes"],
                cbar=False, annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(f"{name}\nTest AUC = {auc:.4f}",
                 fontsize=11, fontweight="bold", color=C["grey_dark"])
    ax.tick_params(labelsize=9)

plt.tight_layout()
save_fig("fig_01_confusion_matrices.png")

# ── Fig 02: ROC curves — all three models ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for name, proba, col in [
    ("Logistic Regression", lr_proba,  C["steel_dark"]),
    ("Random Forest",        rf_proba,  C["positive"]),
    ("XGBoost",              xgb_proba, C["maroon"]),
]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{name}  (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random  (AUC = 0.500)")
ax.set_title("ROC Curves — Recommendation Propensity",
             fontsize=13, fontweight="bold", color=C["grey_dark"])
ax.set_xlabel("False Positive Rate", fontsize=10)
ax.set_ylabel("True Positive Rate", fontsize=10)
ax.legend(loc="lower right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
save_fig("fig_02_roc_curves.png")

# ── Fig 03: Model performance bar chart (CV vs Test AUC) ─────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
model_names = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
cv_means = [lr_cv.mean(), rf_cv.mean(), xgb_cv.mean()]
cv_stds  = [lr_cv.std(),  rf_cv.std(),  xgb_cv.std()]
test_aucs = [lr_auc, rf_auc, xgb_auc]

x, w = np.arange(3), 0.35
b_cv   = ax.bar(x - w/2, cv_means,  w, yerr=cv_stds, capsize=5,
                color=C["steel_mid"], alpha=0.85, label="5-fold CV AUC")
b_test = ax.bar(x + w/2, test_aucs, w,
                color=C["maroon"],   alpha=0.90, label="Test AUC")
for bar, val in zip(b_cv,   cv_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
            f"{val:.3f}", ha="center", fontsize=9, color=C["steel_dark"])
for bar, val in zip(b_test, test_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
            f"{val:.3f}", ha="center", fontsize=9, fontweight="bold", color=C["maroon"])
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("ROC-AUC", fontsize=10)
ax.set_title("Model Performance — CV AUC vs Test AUC",
             fontsize=13, fontweight="bold", color=C["grey_dark"])
ax.legend(fontsize=9, loc="lower right")
ax.axhline(0.5, color="grey", ls="--", lw=1, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
save_fig("fig_03_model_performance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Feature analysis charts...")

feature_names_display = [clean_name(c) for c in X_df.columns]

# ── Fig 04: XGBoost feature importance — top 20 ───────────────────────────────
xgb_imp  = pd.Series(xgb_m.feature_importances_, index=X_df.columns)
xgb_top20 = xgb_imp.nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
bar_colors = [
    C["maroon"] if col.startswith("feat_") else C["steel_mid"]
    for col in xgb_top20.index
]
bars = ax.barh([clean_name(c) for c in xgb_top20.index], xgb_top20.values,
               color=bar_colors, height=0.7)
for bar, val in zip(bars, xgb_top20.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8.5, color=C["grey_dark"])
ax.set_xlabel("Feature Importance (XGBoost)", fontsize=10)
ax.set_title("Top 20 Features — Recommendation Propensity (XGBoost)",
             fontsize=13, fontweight="bold", color=C["grey_dark"], pad=12)
ax.legend(handles=[
    mpatches.Patch(color=C["maroon"],    label="Product / NLP feature"),
    mpatches.Patch(color=C["steel_mid"], label="Categorical / temporal"),
], loc="lower right", fontsize=9)
ax.set_xlim(0, xgb_top20.max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
save_fig("fig_04_feature_importance_xgboost.png")

# ── Fig 05: Random Forest feature importance — top 20 ────────────────────────
rf_imp   = pd.Series(rf.feature_importances_, index=X_df.columns)
rf_top20 = rf_imp.nlargest(20).sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
bar_colors_rf = [
    C["maroon"] if col.startswith("feat_") else C["steel_mid"]
    for col in rf_top20.index
]
bars_rf = ax.barh([clean_name(c) for c in rf_top20.index], rf_top20.values,
                  color=bar_colors_rf, height=0.7)
for bar, val in zip(bars_rf, rf_top20.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8.5, color=C["grey_dark"])
ax.set_xlabel("Feature Importance (Random Forest — Mean Decrease Impurity)", fontsize=10)
ax.set_title("Top 20 Features — Recommendation Propensity (Random Forest)",
             fontsize=13, fontweight="bold", color=C["grey_dark"], pad=12)
ax.legend(handles=[
    mpatches.Patch(color=C["maroon"],    label="Product / NLP feature"),
    mpatches.Patch(color=C["steel_mid"], label="Categorical / temporal"),
], loc="lower right", fontsize=9)
ax.set_xlim(0, rf_top20.max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
save_fig("fig_05_feature_importance_rf.png")

# ── Fig 06: Logistic regression coefficients ──────────────────────────────────
lr_coef = pd.Series(lr.coef_[0], index=X_df.columns)
lr_display = pd.concat([lr_coef.nsmallest(10), lr_coef.nlargest(10)]).sort_values()

fig, ax = plt.subplots(figsize=(11, 8))
bar_colors_lr = [C["negative"] if v < 0 else C["positive"] for v in lr_display.values]
ax.barh([clean_name(c) for c in lr_display.index], lr_display.values,
        color=bar_colors_lr, height=0.7)
ax.axvline(0, color=C["grey_dark"], lw=0.9)
ax.set_xlabel("Coefficient (standardised features — log-odds scale)", fontsize=10)
ax.set_title("Recommendation Drivers — Logistic Regression Coefficients",
             fontsize=13, fontweight="bold", color=C["grey_dark"], pad=12)
ax.legend(handles=[
    mpatches.Patch(color=C["positive"], label="Pushes toward Recommend"),
    mpatches.Patch(color=C["negative"], label="Pushes toward Not Recommend"),
], fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
save_fig("fig_06_lr_coefficients.png")

# ── Fig 07: SHAP summary (XGBoost) ───────────────────────────────────────────
print("  → Computing SHAP values (this may take ~30s)...")
explainer   = shap.TreeExplainer(xgb_m)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(
    shap_values, X_test,
    feature_names=feature_names_display,
    max_display=20,
    show=False,
    plot_size=(10, 8),
)
plt.title("SHAP Feature Contributions — XGBoost Propensity Model\n"
          "Colour = feature value (red=high, blue=low)  |  x-axis = SHAP impact on output",
          fontsize=11, fontweight="bold", color=C["grey_dark"], pad=14)
plt.tight_layout()
save_fig("fig_07_shap_summary.png")

# ── Fig 08: Partial dependence — top 5 product features ──────────────────────
# Use XGBoost importance to rank; take top 5 from feat_ namespace (product levers)
top5_feat = xgb_imp[xgb_imp.index.str.startswith("feat_")].nlargest(5).index.tolist()
top5_idx  = [list(X_df.columns).index(f) for f in top5_feat]

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle(
    "Partial Dependence — How Each Feature Shapes Recommendation Probability",
    fontsize=13, fontweight="bold", color=C["grey_dark"], y=1.02,
)
PartialDependenceDisplay.from_estimator(
    rf, X_train, features=top5_idx,
    feature_names=feature_names_display,
    kind="average", ax=axes,
    line_kw={"color": C["maroon"], "lw": 2.5},
    pd_line_kw={"color": C["maroon"]},
)
for i, ax in enumerate(axes):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Recommendation\nProbability" if i == 0 else "", fontsize=8)
    ax.tick_params(labelsize=8)
plt.tight_layout()
save_fig("fig_08_partial_dependence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PRODUCT INVESTMENT FRAMEWORK & OUTPUT FILES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Building investment framework & saving outputs...")

# Plain-English product implications mapped to feature names
_implications = {
    "feat_overall_rating":       ("Overall Rating (aggregated outcome)",
                                   "Reflects all product dimensions — not a direct lever. "
                                   "Decompose into component scores for actionable targets."),
    "feat_value_for_money":      ("Value for Money Score",
                                   "Strongest product lever. Pricing perception must be anchored "
                                   "by tangible product quality — especially in Business class."),
    "feat_food_beverage":        ("Food & Beverage Score",
                                   "Highest-ROI physical product investment. Business class gap "
                                   "between seat comfort (strong) and food (weak) is the single "
                                   "largest addressable quality mismatch."),
    "feat_seat_comfort":         ("Seat Comfort Score",
                                   "Long-haul loyalty driver — particularly for business travellers. "
                                   "Fleet standardisation to QSuites across routes would close the gap."),
    "feat_cabin_staff":          ("Cabin Staff Score",
                                   "High leverage at low marginal cost. Service training "
                                   "programmes deliver outsized NPS impact relative to investment."),
    "feat_ife":                  ("IFE Score",
                                   "Content refresh cadence gap. A quarterly content cycle (vs "
                                   "bi-annual) would address the recurring mid-year dip in scores."),
    "feat_ground_service":       ("Ground Service Score",
                                   "Airport touchpoints set expectations before boarding. "
                                   "High-value origin markets with low ground scores are churn risks."),
    "feat_wifi":                 ("Wi-Fi Score",
                                   "46% non-response rate masks true dissatisfaction. "
                                   "Connectivity investment targets premium and business travellers."),
    "feat_sentiment_score":      ("NLP Sentiment Score",
                                   "Cross-validates product scores. A high product score with low "
                                   "sentiment signals expectation mismatch — not just poor execution."),
    "feat_rebook":               ("Would Rebook Signal",
                                   "Leading churn indicator. Passengers who signal 'maybe' despite "
                                   "recommending are captive advocates — first to switch if a "
                                   "competitor enters the route."),
    "feat_theme_food_beverage":  ("NLP Theme: Food & Beverage",
                                   "Free-text NLP corroborates score data — food themes appear "
                                   "disproportionately in negative reviews, validating score findings."),
    "feat_theme_value":          ("NLP Theme: Value",
                                   "Value perception surfaces in free text even when the Value "
                                   "for Money score is moderate — a leading indicator of score decay."),
    "feat_competitor_mentioned": ("Competitor Mentioned",
                                   "Passengers naming competitors score lower recommendation intent. "
                                   "A live competitive risk signal for route-level commercial teams."),
    "feat_is_sarcastic":         ("Sarcastic Tone (NLP)",
                                   "Sarcasm is a strong negative signal — irony surfaces frustration "
                                   "that star ratings understate. High sarcasm reviews predict churn."),
    "feat_theme_cabin_crew":     ("NLP Theme: Cabin Crew",
                                   "Positive crew mentions in free text strongly predict recommendation "
                                   "— more so than crew scores alone. Training ROI is measurable via NLP."),
    "feat_trip_verified":        ("Trip Verified",
                                   "Verified reviews score 1.1pts lower on average — "
                                   "methodological anchor for interpreting raw Skytrax ratings."),
}

# Build the ranked investment framework table
fi_df = pd.DataFrame({
    "feature":        list(X_df.columns),
    "xgb_importance": xgb_m.feature_importances_,
    "rf_importance":  rf.feature_importances_,
    "lr_coefficient": lr.coef_[0],
})
fi_df["feature_label"]       = fi_df["feature"].apply(clean_name)
fi_df["direction"]           = fi_df["lr_coefficient"].apply(
    lambda v: "Increases recommendation" if v > 0 else "Decreases recommendation"
)
fi_df["product_dimension"]   = fi_df["feature"].map(
    {k: v[0] for k, v in _implications.items()}
)
fi_df["investment_insight"]  = fi_df["feature"].map(
    {k: v[1] for k, v in _implications.items()}
)
fi_df = fi_df.sort_values("xgb_importance", ascending=False).reset_index(drop=True)
fi_df.insert(0, "rank", range(1, len(fi_df) + 1))

fi_df.to_csv(os.path.join(MODEL_DIR, "qatar_feature_importance.csv"), index=False)
fi_df.to_csv(os.path.join(NLP_OUT,   "qatar_feature_importance.csv"), index=False)
print("    ✓ qatar_feature_importance.csv")

# ── Fig 09: Product investment framework visual ────────────────────────────────
prod_fi = fi_df[fi_df["feature"].str.startswith("feat_")].head(12).copy()
prod_fi_plot = prod_fi.iloc[::-1]  # reverse for top-to-bottom bar chart

fig, ax = plt.subplots(figsize=(11, 7))
bar_cols = [
    C["positive"] if d == "Increases recommendation" else C["negative"]
    for d in prod_fi_plot["direction"]
]
ax.barh(prod_fi_plot["feature_label"], prod_fi_plot["xgb_importance"] * 100,
        color=bar_cols, height=0.65, alpha=0.88)
ax.set_xlabel("Feature Importance Score (XGBoost × 100)", fontsize=10)
ax.set_title(
    "Product Investment Prioritisation Framework\n"
    "Top 12 Product & NLP Features Ranked by Impact on Recommendation Propensity",
    fontsize=12, fontweight="bold", color=C["grey_dark"], pad=12,
)
ax.legend(handles=[
    mpatches.Patch(color=C["positive"], label="Increases recommendation probability"),
    mpatches.Patch(color=C["negative"], label="Decreases recommendation probability"),
], fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(C["grey_light"])
ax.tick_params(labelsize=9.5)
plt.tight_layout()
save_fig("fig_09_investment_framework.png")

# ── Results CSV — one row per test passenger ──────────────────────────────────
results = pd.DataFrame({
    "actual":        y_test,
    "lr_predicted":  lr_pred,
    "lr_proba":      np.round(lr_proba,  4),
    "rf_predicted":  rf_pred,
    "rf_proba":      np.round(rf_proba,  4),
    "xgb_predicted": xgb_pred,
    "xgb_proba":     np.round(xgb_proba, 4),
})
results.to_csv(os.path.join(MODEL_DIR, "qatar_propensity_results.csv"), index=False)
results.to_csv(os.path.join(NLP_OUT,   "qatar_propensity_results.csv"), index=False)
print(f"    ✓ qatar_propensity_results.csv ({len(results):,} rows)")

# ── Save trained models ───────────────────────────────────────────────────────
bundle = {
    "xgb":           xgb_m,
    "rf":            rf,
    "lr":            lr,
    "scaler":        scaler,
    "imputer":       imputer,
    "feature_names": list(X_df.columns),
}
model_path = os.path.join(MODEL_DIR, "qatar_propensity_model.pkl")
with open(model_path, "wb") as fh:
    pickle.dump(bundle, fh)
print(f"    ✓ qatar_propensity_model.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  CLASSIFICATION REPORTS")
print("=" * 62)
for name, pred in [
    ("Logistic Regression", lr_pred),
    ("Random Forest",        rf_pred),
    ("XGBoost",              xgb_pred),
]:
    print(f"\n── {name} ──")
    print(classification_report(y_test, pred,
                                  target_names=["Not Recommend", "Recommend"]))

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("  MODEL SUMMARY")
print("=" * 62)
print(f"\n  {'Model':<25} {'CV AUC':>10}  {'±':>6}  {'Test AUC':>10}")
print(f"  {'-' * 55}")
for name, cv_s, t_auc in [
    ("Logistic Regression", lr_cv,  lr_auc),
    ("Random Forest",        rf_cv,  rf_auc),
    ("XGBoost",              xgb_cv, xgb_auc),
]:
    print(f"  {name:<25} {cv_s.mean():>10.4f}  {cv_s.std():>6.4f}  {t_auc:>10.4f}")

print(f"\n  TOP 10 FEATURES (XGBoost importance — product levers only)")
print(f"  {'Rank':<5} {'Feature':<32} {'XGB Imp':>8}  {'Direction'}")
print(f"  {'-' * 70}")
prod_rows = fi_df[fi_df["feature"].str.startswith("feat_")].head(10)
for i, (_, row) in enumerate(prod_rows.iterrows(), 1):
    label = clean_name(row["feature"])[:31]
    direction = "↑ Recommend" if row["lr_coefficient"] > 0 else "↓ Not recommend"
    print(f"  {i:<5} {label:<32} {row['xgb_importance']:>8.4f}  {direction}")

print(f"\n  Charts  → {CHART_DIR}")
print(f"  Outputs → {MODEL_DIR}")
print("\n  Done.\n")
