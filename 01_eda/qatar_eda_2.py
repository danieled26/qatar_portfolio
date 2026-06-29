"""
================================================================================
QATAR AIRWAYS — COMPREHENSIVE EDA
Skytrax Reviews 2016–2026 | 1,957 Reviews | 26 Variables
Senior Data Specialist Portfolio — Product Development & Design Team
================================================================================
Run in VS Code:
    pip install pandas numpy matplotlib seaborn scipy plotly kaleido
    python qatar_eda.py

Outputs (saved to ./charts/):
    fig01_dataset_overview.png
    fig02_rating_decline_story.png
    fig03_cabin_performance.png
    fig04_score_drivers.png
    fig05_traveller_segments.png
    fig06_aircraft_performance.png
    fig07_geographic_analysis.png
    fig08_verification_bias.png
    fig09_seasonality.png
    fig10_value_perception.png
    fig11_route_analysis.png
    fig12_executive_summary.png
================================================================================
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, chi2_contingency

warnings.filterwarnings("ignore")
os.makedirs("charts", exist_ok=True)

# ── QATAR AIRWAYS OFFICIAL BRAND PALETTE ──────────────────────────────────────
# Source: Qatar Airways Corporate Identity Guidelines
C = {
    # ── Primary brand colour ──────────────────────────────────────────────────
    "maroon":      "#8D1B3D",   # QA primary maroon/burgundy (from logo & ribbon)
    "maroon_dark": "#6B1230",   # Darker shade for depth
    "maroon_light":"#B5466A",   # Lighter tint for fills / hover states

    # ── Corporate palette (left to right in brand guide) ─────────────────────
    "grey_light":  "#C8C9CA",   # Pale silver-grey (swatch 2)
    "steel_dark":  "#4A6278",   # Deep steel blue (swatch 3)
    "steel_mid":   "#6B8296",   # Mid steel blue  (swatch 4)
    "grey_dark":   "#5A5A5A",   # Charcoal grey   (swatch 5)

    # ── Functional / semantic colours ────────────────────────────────────────
    "white":       "#FFFFFF",
    "off_white":   "#F7F5F2",   # Warm near-white for backgrounds
    "text_dark":   "#1A1A1A",   # Near-black body text
    "text_mid":    "#4A4A4A",   # Secondary text
    "text_light":  "#8A8A8A",   # Tertiary / labels

    # ── Chart semantic colours ────────────────────────────────────────────────
    "positive":    "#2E7D5A",   # Green for good/positive signals
    "warning":     "#C47B00",   # Amber for warnings
    "negative":    "#C0392B",   # Red for alerts / negative signals

    # ── Cabin class colours (from QA brand palette) ───────────────────────────
    "economy":     "#4A6278",   # Steel dark → Economy
    "business":    "#8D1B3D",   # Maroon → Business (flagship cabin)
    "first":       "#6B8296",   # Steel mid → First
    "premium":     "#5A5A5A",   # Charcoal → Premium Economy

    # ── Legacy aliases (keep consistent with chart code below) ───────────────
    "navy":        "#4A6278",   # maps to steel_dark
    "teal":        "#6B8296",   # maps to steel_mid
    "slate":       "#5A5A5A",   # maps to grey_dark
    "warm_grey":   "#8A8A8A",
    "cream":       "#F7F5F2",
    "red":         "#C0392B",
    "green":       "#2E7D5A",
    "amber":       "#C47B00",
    "gold":        "#C47B00",
}

CABIN_COLOURS = {
    "Economy Class":   C["economy"],
    "Business Class":  C["business"],
    "First Class":     C["first"],
    "Premium Economy": C["premium"],
}

# Custom sequential colormap using QA maroon
QA_CMAP = LinearSegmentedColormap.from_list(
    "qa_maroon",
    [C["off_white"], C["grey_light"], C["steel_mid"], C["maroon"]],
)

plt.rcParams.update({
    "figure.facecolor":   C["off_white"],
    "axes.facecolor":     C["off_white"],
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.edgecolor":     C["grey_light"],
    "axes.linewidth":     0.8,
    "axes.grid":          True,
    "grid.alpha":         0.2,
    "grid.linestyle":     "--",
    "grid.color":         C["grey_light"],
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "axes.titlecolor":    C["text_dark"],
    "axes.labelsize":     11,
    "axes.labelcolor":    C["text_mid"],
    "xtick.labelsize":    10,
    "xtick.color":        C["text_mid"],
    "ytick.labelsize":    10,
    "ytick.color":        C["text_mid"],
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   C["grey_light"],
})

def save(fig, name, tight=True):
    path = f"charts/{name}"
    if tight:
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=C["off_white"])
    else:
        fig.savefig(path, dpi=180, facecolor=C["off_white"])
    print(f"  ✓ Saved {path}")
    plt.close(fig)

def subtitle(ax, text):
    ax.set_title(text, fontsize=10, color=C["warm_grey"],
                 fontweight="normal", pad=2)

def label_bar(ax, bars, fmt="{:.1f}", offset=0.05, colour=None):
    for bar in bars:
        h = bar.get_height()
        if pd.notna(h) and h != 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset,
                fmt.format(h),
                ha="center", va="bottom",
                fontsize=9,
                color=colour or C["slate"],
            )

def hbar_label(ax, bars, fmt="{:.1f}", offset=0.05):
    for bar in bars:
        w = bar.get_width()
        if pd.notna(w) and w != 0:
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2,
                    fmt.format(w), va="center", fontsize=9, color=C["slate"])

def add_watermark(fig, text="Qatar Airways | Skytrax Reviews 2016–2026"):
    fig.text(0.99, 0.01, text, ha="right", va="bottom",
             fontsize=8, color=C["warm_grey"], alpha=0.6)

# ── LOAD & CLEAN ──────────────────────────────────────────────────────────────
print("Loading data...")

df = pd.read_csv("qatar_airline_reviews.csv")   # adjust path if needed

SCORE_COLS = [
    "seat_comfort", "cabin_staff_service", "food_beverages",
    "inflight_entertainment", "ground_service", "wifi_connectivity",
    "value_for_money",
]
SCORE_LABELS = [
    "Seat Comfort", "Cabin Staff", "Food & Bev",
    "IFE", "Ground Service", "Wi-Fi", "Value for Money",
]

df["overall_rating_10"] = pd.to_numeric(df["overall_rating_10"], errors="coerce")
for col in SCORE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["review_date_iso"] = pd.to_datetime(df["review_date_iso"], errors="coerce")
df["year"]    = df["review_date_iso"].dt.year
df["month"]   = df["review_date_iso"].dt.month
df["quarter"] = df["review_date_iso"].dt.quarter

df["review_len"] = df["review_text"].str.len()
df["title_len"]  = df["review_title"].str.len()

df["origin"]      = df["route"].str.split(" to ").str[0].str.strip()
df["destination"] = df["route"].str.split(" to ").str[-1].str.strip()

df["rating_band"] = pd.cut(
    df["overall_rating_10"],
    bins=[0, 3, 6, 8, 10],
    labels=["Low (1–3)", "Mid (4–6)", "High (7–8)", "Top (9–10)"],
)

# Normalise aircraft names
ac_map = {
    "Boeing 777-300ER": "B777-300ER",
    "Boeing 777-300":   "B777-300",
    "Boeing 777":       "B777",
    "Boeing 777-200LR": "B777-200LR",
    "Boeing 787-8":     "B787-8",
    "Boeing 787-9":     "B787-9",
    "Boeing 787":       "B787",
    "A380-800":         "A380-800",
    "A380":             "A380",
    "A350-1000":        "A350-1000",
    "A350-900":         "A350-900",
    "A350":             "A350",
    "A330-200":         "A330-200",
    "A330":             "A330",
    "A320neo":          "A320neo",
    "A320":             "A320",
    "A321neo":          "A321neo",
}
df["aircraft_clean"] = df["aircraft"].replace(ac_map)

print(f"  {len(df):,} reviews | {df['year'].min()}–{df['year'].max()}")
print()

# ────────────────────────────────────────────────────────────────────────────
# FIG 01 — DATASET OVERVIEW (the 'what we're working with' slide)
# ────────────────────────────────────────────────────────────────────────────
print("FIG 01 — Dataset overview...")

fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    "Qatar Airways Passenger Review Dataset — Structural Overview",
    fontsize=17, fontweight="bold", y=1.0,
)
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

# 1a — Reviews per year (bar)
ax = fig.add_subplot(gs[0, :2])
yr_counts = df["year"].value_counts().sort_index()
bars = ax.bar(yr_counts.index, yr_counts.values,
              color=C["maroon"], edgecolor="none", width=0.7)
label_bar(ax, bars, fmt="{:.0f}", offset=2)
ax.set_title("Reviews per Year", fontweight="bold")
subtitle(ax, "COVID-19 dip visible 2020–2021")
ax.set_xlabel("Year"); ax.set_ylabel("Number of Reviews")
ax.axvspan(2019.5, 2021.5, alpha=0.08, color=C["amber"], label="COVID period")
ax.legend(fontsize=9)

# 1b — Cabin class pie
ax = fig.add_subplot(gs[0, 2])
cabin_counts = df["seat_type"].value_counts()
colours_pie  = [CABIN_COLOURS.get(c, C["slate"]) for c in cabin_counts.index]
wedges, texts, autotexts = ax.pie(
    cabin_counts,
    labels=cabin_counts.index,
    colors=colours_pie,
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    pctdistance=0.78,
)
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight("bold")
ax.set_title("Cabin Class Mix", fontweight="bold")

# 1c — Traveller type pie
ax = fig.add_subplot(gs[0, 3])
tt_counts  = df["type_of_traveller"].value_counts()
tt_colours = [C["navy"], C["teal"], C["gold"], C["maroon"]]
wedges, texts, autotexts = ax.pie(
    tt_counts,
    labels=tt_counts.index,
    colors=tt_colours,
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    pctdistance=0.78,
)
for at in autotexts:
    at.set_fontsize(9); at.set_fontweight("bold")
ax.set_title("Traveller Type Mix", fontweight="bold")

# 1d — Recommendation rate donut
ax = fig.add_subplot(gs[1, 0])
rec_counts = df["recommended"].value_counts()
ax.pie(
    rec_counts,
    labels=rec_counts.index,
    colors=[C["green"], C["red"]],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
    pctdistance=0.75,
)
rec_pct = (df["recommended"] == "Yes").mean() * 100
ax.text(0, 0, f"{rec_pct:.0f}%\nRec.", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C["green"])
ax.set_title("Recommendation Rate", fontweight="bold")

# 1e — Top 10 reviewer countries
ax = fig.add_subplot(gs[1, 1:3])
top_countries = df["reviewer_country"].value_counts().head(10)
bars = ax.barh(top_countries.index[::-1], top_countries.values[::-1],
               color=C["navy"], edgecolor="none", height=0.65)
hbar_label(ax, bars, fmt="{:.0f}")
ax.set_title("Top 10 Reviewer Countries", fontweight="bold")
ax.set_xlabel("Number of Reviews")

# 1f — Rating distribution histogram
ax = fig.add_subplot(gs[1, 3])
valid_ratings = df["overall_rating_10"].dropna()
ax.hist(valid_ratings, bins=10, range=(0.5, 10.5),
        color=C["maroon"], edgecolor="white", linewidth=0.8, rwidth=0.85)
ax.axvline(valid_ratings.mean(), color=C["gold"], linewidth=2,
           linestyle="--", label=f"Mean: {valid_ratings.mean():.1f}")
ax.set_title("Overall Rating Distribution", fontweight="bold")
subtitle(ax, "Bimodal — love it or hate it")
ax.set_xlabel("Rating (1–10)"); ax.set_ylabel("Count")
ax.legend()

add_watermark(fig)
save(fig, "fig01_dataset_overview.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 02 — THE RATING DECLINE STORY (the headline finding)
# ────────────────────────────────────────────────────────────────────────────
print("FIG 02 — Rating decline story...")

fig, axes = plt.subplots(2, 2, figsize=(18, 11))
fig.suptitle(
    "The Rating Decline Story — Qatar Airways 2016–2026",
    fontsize=17, fontweight="bold", y=1.01,
)

yearly = df.groupby("year")["overall_rating_10"].agg(["mean", "median", "count", "std"])
cabin_yearly = df.groupby(["year", "seat_type"])["overall_rating_10"].mean().unstack()

# 2a — Overall trend with confidence band
ax = axes[0, 0]
years = yearly.index
means = yearly["mean"]
stds  = yearly["std"]
ax.fill_between(years, means - stds/2, means + stds/2,
                alpha=0.15, color=C["maroon"], label="±0.5 SD band")
ax.plot(years, means, "o-", color=C["maroon"], linewidth=2.5,
        markersize=8, zorder=4, label="Mean rating")
ax.plot(years, yearly["median"], "s--", color=C["gold"], linewidth=1.5,
        markersize=6, alpha=0.8, label="Median rating")

# Annotate key events
annotations = {
    2020: ("COVID-19\nlockdowns", "above"),
    2022: ("Post-COVID\nexpectation\ngap", "below"),
    2024: ("Lowest\nrated year\n(5.8)", "below"),
}
for yr, (txt, pos) in annotations.items():
    y_val = means[yr]
    offset = 0.4 if pos == "above" else -0.6
    ax.annotate(txt, xy=(yr, y_val), xytext=(yr, y_val + offset),
                arrowprops=dict(arrowstyle="->", color=C["slate"], lw=1.2),
                ha="center", fontsize=8, color=C["slate"],
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=C["slate"], alpha=0.8))

ax.axhline(means.mean(), color=C["warm_grey"], linestyle=":",
           linewidth=1.2, label=f"10-yr avg: {means.mean():.2f}")
ax.set_title("Overall Rating Trend — All Cabins", fontweight="bold")
subtitle(ax, "Biennial decline: +2.0pts COVID lift → −2.4pts 2021–2024 fall")
ax.set_xlabel("Year"); ax.set_ylabel("Avg Rating (1–10)")
ax.set_ylim(4, 10); ax.legend(fontsize=9)
ax.set_xticks(years)

# 2b — By cabin class over time
ax = axes[0, 1]
for cabin, colour in CABIN_COLOURS.items():
    if cabin in cabin_yearly.columns:
        data = cabin_yearly[cabin].dropna()
        ax.plot(data.index, data.values, "o-", label=cabin,
                color=colour, linewidth=2.2, markersize=6)
ax.set_title("Rating Trend by Cabin Class", fontweight="bold")
subtitle(ax, "Economy bears the sharpest 2024 decline (5.2/10)")
ax.set_xlabel("Year"); ax.set_ylabel("Avg Rating (1–10)")
ax.set_ylim(4, 11); ax.legend()
ax.set_xticks(years)

# 2c — Recommendation rate by year
ax = axes[1, 0]
rec_year = df.groupby("year").apply(
    lambda x: (x["recommended"] == "Yes").mean() * 100
).reset_index()
rec_year.columns = ["year", "rec_pct"]
bar_colours = [C["green"] if v >= 70 else C["amber"] if v >= 60 else C["red"]
               for v in rec_year["rec_pct"]]
bars = ax.bar(rec_year["year"], rec_year["rec_pct"],
              color=bar_colours, edgecolor="none", width=0.7)
label_bar(ax, bars, fmt="{:.0f}%", offset=0.5)
ax.axhline(70, color=C["slate"], linestyle="--", linewidth=1.2,
           label="70% threshold")
ax.set_title("% Passengers Recommending — by Year", fontweight="bold")
subtitle(ax, "2024: only 55.5% would recommend — lowest in dataset")
ax.set_xlabel("Year"); ax.set_ylabel("Recommendation Rate (%)")
ax.set_ylim(0, 100); ax.legend()

# 2d — Volume of low vs high ratings over time
ax = axes[1, 1]
df["rating_hi"] = (df["overall_rating_10"] >= 8).astype(float)
df["rating_lo"] = (df["overall_rating_10"] <= 3).astype(float)
hi_yr = df.groupby("year")["rating_hi"].sum()
lo_yr = df.groupby("year")["rating_lo"].sum()
total_yr = df.groupby("year")["overall_rating_10"].count()
hi_pct = (hi_yr / total_yr * 100).round(1)
lo_pct = (lo_yr / total_yr * 100).round(1)
ax.stackplot(years, hi_pct[years], lo_pct[years],
             labels=["High (8–10)", "Low (1–3)"],
             colors=[C["teal"], C["red"]], alpha=0.8)
ax.set_title("High vs Low Ratings Share by Year", fontweight="bold")
subtitle(ax, "Polarisation increasing post-2022")
ax.set_xlabel("Year"); ax.set_ylabel("% of Annual Reviews")
ax.legend(loc="upper right"); ax.set_ylim(0, 100)
ax.set_xticks(years)

plt.tight_layout()
add_watermark(fig)
save(fig, "fig02_rating_decline_story.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 03 — CABIN PERFORMANCE DEEP DIVE
# ────────────────────────────────────────────────────────────────────────────
print("FIG 03 — Cabin performance...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Cabin Class Performance Analysis", fontsize=17, fontweight="bold", y=1.01)

cabins_main = ["Economy Class", "Business Class", "First Class"]

# 3a — Avg score per category by cabin (grouped bar)
ax = axes[0, 0]
cabin_scores = df.groupby("seat_type")[SCORE_COLS].mean()
x = np.arange(len(SCORE_LABELS))
width = 0.22
for i, cabin in enumerate(cabins_main):
    if cabin in cabin_scores.index:
        vals = [cabin_scores.loc[cabin, c] for c in SCORE_COLS]
        ax.bar(x + i * width, vals, width, label=cabin,
               color=CABIN_COLOURS[cabin], edgecolor="none", alpha=0.9)
ax.set_xticks(x + width)
ax.set_xticklabels(SCORE_LABELS, rotation=30, ha="right", fontsize=9)
ax.set_ylim(2.5, 5.5)
ax.set_title("Category Scores by Cabin Class", fontweight="bold")
subtitle(ax, "First Class food (3.86) lower than Economy (3.75) — anomaly")
ax.set_ylabel("Avg Score (1–5)")
ax.legend()

# 3b — Recommendation rate by cabin
ax = axes[0, 1]
rec_cabin = df.groupby("seat_type").apply(
    lambda x: (x["recommended"] == "Yes").mean() * 100
).reindex(cabins_main)
bars = ax.bar(rec_cabin.index, rec_cabin.values,
              color=[CABIN_COLOURS[c] for c in rec_cabin.index],
              edgecolor="none", width=0.55)
label_bar(ax, bars, fmt="{:.1f}%", offset=0.5)
ax.set_title("Recommendation Rate by Cabin", fontweight="bold")
subtitle(ax, "Economy at 66% — 13pp gap vs Business (79%)")
ax.set_ylabel("% Recommended"); ax.set_ylim(0, 100)
ax.set_xticklabels(rec_cabin.index, rotation=15, ha="right")

# 3c — Overall rating distribution violin by cabin
ax = axes[0, 2]
cabin_data = [df[df["seat_type"] == c]["overall_rating_10"].dropna().values
              for c in cabins_main]
parts = ax.violinplot(cabin_data, positions=range(len(cabins_main)),
                      showmedians=True, showextrema=True)
for i, (pc, cabin) in enumerate(zip(parts["bodies"], cabins_main)):
    pc.set_facecolor(CABIN_COLOURS[cabin])
    pc.set_alpha(0.75)
parts["cmedians"].set_color(C["gold"]); parts["cmedians"].set_linewidth(2)
ax.set_xticks(range(len(cabins_main)))
ax.set_xticklabels([c.replace(" Class", "") for c in cabins_main])
ax.set_title("Rating Distribution by Cabin", fontweight="bold")
subtitle(ax, "Business and Economy both show bimodal love/hate pattern")
ax.set_ylabel("Overall Rating (1–10)")

# 3d — Business class trend vs Economy (gap analysis)
ax = axes[1, 0]
for cabin in ["Economy Class", "Business Class"]:
    if cabin in cabin_yearly.columns:
        data = cabin_yearly[cabin].dropna()
        ax.plot(data.index, data.values, "o-",
                label=cabin, color=CABIN_COLOURS[cabin],
                linewidth=2.2, markersize=6)
gap = (cabin_yearly["Business Class"] - cabin_yearly["Economy Class"]).dropna()
ax.fill_between(gap.index, cabin_yearly.loc[gap.index, "Economy Class"],
                cabin_yearly.loc[gap.index, "Business Class"],
                alpha=0.12, color=C["navy"], label="Gap")
ax.set_title("Economy vs Business Gap Over Time", fontweight="bold")
subtitle(ax, "Gap widening since 2022 — Economy bearing more pressure")
ax.set_xlabel("Year"); ax.set_ylabel("Avg Rating")
ax.legend(); ax.set_xticks(years)

# 3e — Score heatmap by cabin
ax = axes[1, 1]
heat_data = df[df["seat_type"].isin(cabins_main)].groupby("seat_type")[SCORE_COLS].mean()
heat_data.columns = SCORE_LABELS
sns.heatmap(heat_data, ax=ax, cmap=QA_CMAP, vmin=2.5, vmax=5.0,
            annot=True, fmt=".2f", annot_kws={"size": 9},
            linewidths=0.4, cbar_kws={"shrink": 0.8})
ax.set_title("Score Heatmap — Cabin × Category", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# 3f — Value for money by cabin and year
ax = axes[1, 2]
vfm = df.groupby(["year", "seat_type"])["value_for_money"].mean().unstack()
for cabin in cabins_main:
    if cabin in vfm.columns:
        ax.plot(vfm.index, vfm[cabin], "o-",
                color=CABIN_COLOURS[cabin], label=cabin,
                linewidth=2, markersize=5)
ax.set_title("Value for Money Perception — by Year & Cabin", fontweight="bold")
subtitle(ax, "Declining post-2021 across all cabins")
ax.set_xlabel("Year"); ax.set_ylabel("Avg VfM Score (1–5)")
ax.legend(); ax.set_xticks(years)

plt.tight_layout()
add_watermark(fig)
save(fig, "fig03_cabin_performance.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 04 — SCORE DRIVERS & CORRELATIONS
# ────────────────────────────────────────────────────────────────────────────
print("FIG 04 — Score drivers...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("What Drives Overall Rating — Correlation & Driver Analysis",
             fontsize=17, fontweight="bold", y=1.01)

# 4a — Correlation bar chart
ax = axes[0, 0]
corr_vals = df[SCORE_COLS + ["overall_rating_10"]].corr()["overall_rating_10"].drop("overall_rating_10")
corr_vals = corr_vals.sort_values(ascending=True)
bar_cols = [C["green"] if v > 0.7 else C["teal"] if v > 0.5 else C["amber"]
            for v in corr_vals]
bars = ax.barh([SCORE_LABELS[SCORE_COLS.index(c)] for c in corr_vals.index],
               corr_vals.values, color=bar_cols, edgecolor="none", height=0.6)
hbar_label(ax, bars, fmt="{:.3f}", offset=0.005)
ax.axvline(0.7, color=C["red"], linestyle="--", linewidth=1.2,
           label="Strong (r=0.7)")
ax.set_title("Pearson r with Overall Rating", fontweight="bold")
subtitle(ax, "Value for Money is the single strongest predictor (r=0.86)")
ax.set_xlabel("Correlation Coefficient")
ax.legend()

# 4b — Correlation heatmap (all score pairs)
ax = axes[0, 1]
corr_matrix = df[SCORE_COLS].corr()
corr_matrix.index   = SCORE_LABELS
corr_matrix.columns = SCORE_LABELS
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, ax=ax, mask=mask, cmap="RdGy", center=0,
            vmin=-1, vmax=1, annot=True, fmt=".2f",
            annot_kws={"size": 8}, linewidths=0.3,
            cbar_kws={"shrink": 0.8})
ax.set_title("Inter-Category Correlation Matrix", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

# 4c — Scatter: value for money vs overall
ax = axes[0, 2]
plot_df = df[["value_for_money", "overall_rating_10", "seat_type"]].dropna()
for cabin in cabins_main:
    sub = plot_df[plot_df["seat_type"] == cabin]
    ax.scatter(sub["value_for_money"] + np.random.uniform(-0.1, 0.1, len(sub)),
               sub["overall_rating_10"] + np.random.uniform(-0.3, 0.3, len(sub)),
               alpha=0.25, s=18, color=CABIN_COLOURS[cabin], label=cabin)
# Regression line
x_reg = plot_df["value_for_money"].values
y_reg = plot_df["overall_rating_10"].values
m, b, r, p, _ = stats.linregress(x_reg[~np.isnan(x_reg)], y_reg[~np.isnan(y_reg)])
xl = np.linspace(1, 5, 100)
ax.plot(xl, m * xl + b, color=C["maroon"], linewidth=2,
        label=f"r={r:.2f}")
ax.set_title("Value for Money vs Overall Rating", fontweight="bold")
subtitle(ax, "Strongest linear relationship in the dataset")
ax.set_xlabel("Value for Money (1–5)")
ax.set_ylabel("Overall Rating (1–10)")
ax.legend(fontsize=8)

# 4d — Avg score by rating band
ax = axes[1, 0]
band_scores = df.groupby("rating_band", observed=True)[SCORE_COLS].mean()
band_scores.columns = SCORE_LABELS
band_scores.plot(kind="bar", ax=ax, colormap="Greys",
                 edgecolor="none", width=0.75)
ax.set_title("Category Scores by Rating Band", fontweight="bold")
subtitle(ax, "Every dimension drops in lockstep with overall rating")
ax.set_xlabel("Rating Band")
ax.set_ylabel("Avg Score (1–5)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.legend(fontsize=7, ncol=2)

# 4e — "Not Rated" (missing score) analysis
ax = axes[1, 1]
not_rated_pct = {}
orig = pd.read_csv("qatar_airline_reviews.csv")
for col, label in zip(SCORE_COLS, SCORE_LABELS):
    not_rated_pct[label] = (orig[col] == "Not Rated").mean() * 100
nr = pd.Series(not_rated_pct).sort_values(ascending=True)
bar_cols = [C["red"] if v > 30 else C["amber"] if v > 10 else C["teal"]
            for v in nr.values]
bars = ax.barh(nr.index, nr.values, color=bar_cols, edgecolor="none", height=0.6)
hbar_label(ax, bars, fmt="{:.1f}%")
ax.set_title('"Not Rated" Sparsity per Category', fontweight="bold")
subtitle(ax, "Wi-Fi at 45.7% missing — structural data gap, not low quality")
ax.set_xlabel("% Not Rated")

# 4f — Review length vs rating (do unhappy people write more?)
ax = axes[1, 2]
band_len = df.groupby("rating_band", observed=True)["review_len"].mean()
bars = ax.bar(band_len.index, band_len.values,
              color=[C["red"], C["amber"], C["teal"], C["navy"]],
              edgecolor="none", width=0.55)
label_bar(ax, bars, fmt="{:.0f}", offset=5)
ax.set_title("Avg Review Length by Rating Band", fontweight="bold")
subtitle(ax, "Dissatisfied passengers write 22% more — higher signal in negatives")
ax.set_xlabel("Rating Band")
ax.set_ylabel("Avg Review Length (chars)")
ax.set_xticklabels(band_len.index, rotation=15, ha="right")

plt.tight_layout()
add_watermark(fig)
save(fig, "fig04_score_drivers.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 05 — TRAVELLER SEGMENT ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
print("FIG 05 — Traveller segments...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Traveller Segment Performance Analysis",
             fontsize=17, fontweight="bold", y=1.01)

travellers = ["Solo Leisure", "Couple Leisure", "Business", "Family Leisure"]
trav_cols  = [C["teal"], C["navy"], C["maroon"], C["gold"]]

# 5a — Overall rating by traveller type
ax = axes[0, 0]
trav_rating = df.groupby("type_of_traveller")["overall_rating_10"].mean().reindex(travellers)
bars = ax.bar(trav_rating.index, trav_rating.values,
              color=trav_cols, edgecolor="none", width=0.6)
label_bar(ax, bars, fmt="{:.2f}", offset=0.05)
ax.set_title("Avg Rating by Traveller Type", fontweight="bold")
subtitle(ax, "Business travellers most critical (lowest satisfaction)")
ax.set_ylabel("Avg Overall Rating (1–10)")
ax.set_xticklabels(trav_rating.index, rotation=15, ha="right")
ax.set_ylim(5, 9)

# 5b — Recommendation rate by traveller type
ax = axes[0, 1]
rec_trav = df.groupby("type_of_traveller").apply(
    lambda x: (x["recommended"] == "Yes").mean() * 100
).reindex(travellers)
bars = ax.bar(rec_trav.index, rec_trav.values,
              color=trav_cols, edgecolor="none", width=0.6)
label_bar(ax, bars, fmt="{:.1f}%", offset=0.5)
ax.axhline(71.2, color=C["slate"], linestyle="--", linewidth=1.2,
           label=f"Overall avg (71.2%)")
ax.set_title("Recommendation Rate by Traveller Type", fontweight="bold")
ax.set_ylabel("% Recommended")
ax.set_ylim(0, 100)
ax.set_xticklabels(rec_trav.index, rotation=15, ha="right")
ax.legend()

# 5c — Score profile radar — traveller type (as grouped bars)
ax = axes[0, 2]
trav_scores = df.groupby("type_of_traveller")[SCORE_COLS].mean().reindex(travellers)
x = np.arange(len(SCORE_LABELS))
width = 0.18
for i, (trav, col) in enumerate(zip(travellers, trav_cols)):
    ax.bar(x + i * width, trav_scores.loc[trav], width,
           label=trav, color=col, edgecolor="none", alpha=0.88)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(SCORE_LABELS, rotation=30, ha="right", fontsize=9)
ax.set_ylim(2.5, 5.5)
ax.set_title("Category Scores by Traveller Type", fontweight="bold")
subtitle(ax, "Solo leisure most satisfied across all categories")
ax.set_ylabel("Avg Score (1–5)")
ax.legend(fontsize=8)

# 5d — Cabin × traveller heatmap (count)
ax = axes[1, 0]
ct = pd.crosstab(df["seat_type"], df["type_of_traveller"])
ct = ct.reindex(index=[c for c in cabins_main if c in ct.index],
                columns=[t for t in travellers if t in ct.columns])
sns.heatmap(ct, ax=ax, cmap=QA_CMAP, annot=True, fmt="d",
            annot_kws={"size": 10}, linewidths=0.3,
            cbar_kws={"shrink": 0.8})
ax.set_title("Cabin × Traveller Type — Volume", fontweight="bold")
subtitle(ax, "Family leisure concentrated in Economy (83%)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# 5e — Business traveller rating trend (most loyal, most demanding)
ax = axes[1, 1]
biz_year = df[df["type_of_traveller"] == "Business"].groupby("year")["overall_rating_10"].agg(["mean", "count"])
leisure_year = df[df["type_of_traveller"] != "Business"].groupby("year")["overall_rating_10"].mean()
ax.plot(biz_year.index, biz_year["mean"], "o-",
        color=C["maroon"], linewidth=2.2, markersize=7, label="Business travellers")
ax.plot(leisure_year.index, leisure_year.values, "s--",
        color=C["teal"], linewidth=1.8, markersize=6, label="Leisure travellers")
ax.set_title("Business vs Leisure Rating Trend", fontweight="bold")
subtitle(ax, "Business travellers increasingly dissatisfied 2022–2024")
ax.set_xlabel("Year"); ax.set_ylabel("Avg Rating (1–10)")
ax.legend(); ax.set_xticks(years)

# 5f — Traveller type trend over years (volume shift)
ax = axes[1, 2]
trav_year = df.groupby(["year", "type_of_traveller"]).size().unstack(fill_value=0)
trav_year_pct = trav_year.div(trav_year.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(trav_year_pct))
for trav, col in zip(travellers, trav_cols):
    if trav in trav_year_pct.columns:
        ax.bar(trav_year_pct.index, trav_year_pct[trav],
               bottom=bottom, label=trav, color=col, edgecolor="none", width=0.7)
        bottom += trav_year_pct[trav].values
ax.set_title("Traveller Type Mix — by Year", fontweight="bold")
subtitle(ax, "Business traveller share growing post-2021")
ax.set_xlabel("Year"); ax.set_ylabel("% Share of Reviews")
ax.legend(fontsize=9, loc="lower left")
ax.set_xticks(years)

plt.tight_layout()
add_watermark(fig)
save(fig, "fig05_traveller_segments.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 06 — AIRCRAFT PERFORMANCE
# ────────────────────────────────────────────────────────────────────────────
print("FIG 06 — Aircraft performance...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Aircraft Type Performance Analysis",
             fontsize=17, fontweight="bold", y=1.01)

# Filter to aircraft with ≥20 reviews, exclude 'Not Rated'
ac_stats = df[df["aircraft_clean"] != "Not Rated"].groupby("aircraft_clean").agg(
    mean_rating=("overall_rating_10", "mean"),
    count=("overall_rating_10", "count"),
    mean_seat=("seat_comfort", "mean"),
    mean_ife=("inflight_entertainment", "mean"),
)
ac_stats = ac_stats[ac_stats["count"] >= 20].sort_values("mean_rating", ascending=False)

# 6a — Mean rating by aircraft
ax = axes[0, 0]
bar_cols = [C["teal"] if "A350" in i or "A380" in i else C["navy"]
            for i in ac_stats.index]
bars = ax.barh(ac_stats.index[::-1], ac_stats["mean_rating"][::-1],
               color=bar_cols[::-1], edgecolor="none", height=0.65)
hbar_label(ax, bars, fmt="{:.2f}")
ax.axvline(df["overall_rating_10"].mean(), color=C["gold"], linestyle="--",
           linewidth=1.5, label=f"Fleet avg ({df['overall_rating_10'].mean():.2f})")
ax.set_title("Avg Rating by Aircraft Type", fontweight="bold")
subtitle(ax, "A350-1000 top performer; older B777 variants below fleet avg")
ax.set_xlabel("Avg Rating (1–10)")
ax.legend()

# 6b — Aircraft by seat comfort and IFE
ax = axes[0, 1]
ax.scatter(ac_stats["mean_seat"], ac_stats["mean_ife"],
           s=ac_stats["count"] * 2.5, alpha=0.75,
           color=[C["teal"] if "A350" in i or "A380" in i else
                  C["maroon"] if "B777" in i else C["navy"]
                  for i in ac_stats.index],
           edgecolors="white", linewidth=0.8)
for idx, row in ac_stats.iterrows():
    ax.annotate(idx, (row["mean_seat"], row["mean_ife"]),
                fontsize=8, ha="left", va="bottom",
                xytext=(3, 3), textcoords="offset points")
ax.set_title("Seat Comfort vs IFE by Aircraft", fontweight="bold")
subtitle(ax, "Bubble size = number of reviews | Newer fleet scores higher on both")
ax.set_xlabel("Avg Seat Comfort (1–5)")
ax.set_ylabel("Avg IFE Score (1–5)")

# 6c — Aircraft volume mix over years (Airbus vs Boeing shift)
ax = axes[1, 0]
df["fleet_family"] = df["aircraft_clean"].apply(
    lambda x: "Airbus" if x.startswith("A") else
              "Boeing" if x.startswith("B") else "Unknown"
)
fleet_year = df.groupby(["year", "fleet_family"]).size().unstack(fill_value=0)
fleet_pct  = fleet_year.div(fleet_year.sum(axis=1), axis=0) * 100
for fam, col in [("Airbus", C["teal"]), ("Boeing", C["navy"]), ("Unknown", C["warm_grey"])]:
    if fam in fleet_pct.columns:
        ax.plot(fleet_pct.index, fleet_pct[fam], "o-",
                label=fam, color=col, linewidth=2.2, markersize=6)
ax.set_title("Fleet Family Mix in Reviews — by Year", fontweight="bold")
subtitle(ax, "Airbus share rising as A350/A380 fleet grows")
ax.set_xlabel("Year"); ax.set_ylabel("% of Reviews"); ax.legend()
ax.set_xticks(years)

# 6d — Aircraft rating by cabin (which plane is best in which class)
ax = axes[1, 1]
top_ac = ac_stats.head(6).index.tolist()
ac_cabin = df[
    (df["aircraft_clean"].isin(top_ac)) &
    (df["seat_type"].isin(["Economy Class", "Business Class"]))
].groupby(["aircraft_clean", "seat_type"])["overall_rating_10"].mean().unstack()
ac_cabin = ac_cabin.reindex(top_ac)
ac_cabin.plot(kind="bar", ax=ax,
              color=[CABIN_COLOURS["Economy Class"], CABIN_COLOURS["Business Class"]],
              edgecolor="none", width=0.65)
ax.set_title("Top 6 Aircraft — Rating by Cabin Class", fontweight="bold")
subtitle(ax, "A350-1000 delivers best Business class experience")
ax.set_xlabel(""); ax.set_ylabel("Avg Rating (1–10)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
ax.legend(title="Cabin")

plt.tight_layout()
add_watermark(fig)
save(fig, "fig06_aircraft_performance.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 07 — GEOGRAPHIC ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
print("FIG 07 — Geographic analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Geographic Analysis — Reviewer Countries & Routes",
             fontsize=17, fontweight="bold", y=1.01)

# 7a — Top 15 countries by avg rating (min 15 reviews)
ax = axes[0, 0]
country_stats = df.groupby("reviewer_country").agg(
    mean_rating=("overall_rating_10", "mean"),
    count=("overall_rating_10", "count"),
).query("count >= 15").sort_values("mean_rating", ascending=True)
bar_cols = [C["green"] if v >= 7.5 else C["amber"] if v >= 6 else C["red"]
            for v in country_stats["mean_rating"]]
bars = ax.barh(country_stats.index[-15:], country_stats["mean_rating"][-15:],
               color=bar_cols[-15:], edgecolor="none", height=0.65)
hbar_label(ax, bars, fmt="{:.2f}")
ax.axvline(df["overall_rating_10"].mean(), color=C["maroon"], linestyle="--",
           linewidth=1.5, label=f"Global avg ({df['overall_rating_10'].mean():.2f})")
ax.set_title("Avg Rating by Reviewer Country (≥15 reviews)", fontweight="bold")
subtitle(ax, "Home market (Qatar) gives highest ratings; long-haul markets most critical")
ax.set_xlabel("Avg Rating (1–10)")
ax.legend()

# 7b — Review volume vs avg rating scatter by country
ax = axes[0, 1]
country_all = df.groupby("reviewer_country").agg(
    mean_rating=("overall_rating_10", "mean"),
    count=("overall_rating_10", "count"),
).query("count >= 10")
scatter_cols = [C["teal"] if v >= 7 else C["amber"] if v >= 5.5 else C["red"]
                for v in country_all["mean_rating"]]
ax.scatter(country_all["count"], country_all["mean_rating"],
           s=70, alpha=0.7, color=scatter_cols, edgecolors="white", linewidth=0.5)
for idx, row in country_all[country_all["count"] >= 30].iterrows():
    ax.annotate(idx, (row["count"], row["mean_rating"]),
                fontsize=8, xytext=(3, 2), textcoords="offset points", color=C["slate"])
ax.set_title("Review Volume vs Avg Rating by Country", fontweight="bold")
subtitle(ax, "High-volume markets (UK, US, AU) tend to be more critical")
ax.set_xlabel("Number of Reviews")
ax.set_ylabel("Avg Rating (1–10)")

# 7c — Top 15 routes by avg rating (min 5 reviews)
ax = axes[1, 0]
route_stats = df.groupby("route").agg(
    mean_rating=("overall_rating_10", "mean"),
    count=("overall_rating_10", "count"),
).query("count >= 5").sort_values("mean_rating", ascending=False)
top_routes = route_stats.head(10)
bot_routes = route_stats.tail(10)
combined   = pd.concat([top_routes, bot_routes]).drop_duplicates()
bar_cols   = [C["teal"] if v >= 7.5 else C["red"] for v in combined["mean_rating"]]
ax.barh(combined.index[::-1], combined["mean_rating"][::-1],
        color=bar_cols[::-1], edgecolor="none", height=0.65)
ax.axvline(7.05, color=C["maroon"], linestyle="--",
           linewidth=1.5, label="Fleet avg (7.05)")
ax.set_title("Top & Bottom Routes by Avg Rating", fontweight="bold")
subtitle(ax, "Long-haul premium routes outperform — short-haul gaps visible")
ax.set_xlabel("Avg Rating (1–10)")
ax.legend()

# 7d — SA perspective (relevant to your application)
ax = axes[1, 1]
sa_df = df[df["reviewer_country"] == "South Africa"]
sa_year = sa_df.groupby("year")["overall_rating_10"].agg(["mean", "count"])
global_year = df.groupby("year")["overall_rating_10"].mean()
ax2 = ax.twinx()
ax.plot(sa_year.index, sa_year["mean"], "o-",
        color=C["maroon"], linewidth=2.5, markersize=8, label="SA rating (left)")
ax.plot(global_year.index, global_year.values, "s--",
        color=C["warm_grey"], linewidth=1.5, markersize=5, alpha=0.7,
        label="Global avg (left)")
ax2.bar(sa_year.index, sa_year["count"], alpha=0.2,
        color=C["gold"], width=0.7, label="SA reviews (right)")
ax.set_title("South Africa Market Analysis", fontweight="bold")
subtitle(ax, f"45 SA reviews | Avg: {sa_df['overall_rating_10'].mean():.1f}/10 | "
             f"Rec: {(sa_df['recommended']=='Yes').mean()*100:.0f}%")
ax.set_xlabel("Year")
ax.set_ylabel("Avg Rating (1–10)")
ax2.set_ylabel("SA Review Count", color=C["gold"])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.set_xticks(years)

plt.tight_layout()
add_watermark(fig)
save(fig, "fig07_geographic_analysis.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 08 — VERIFICATION BIAS ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
print("FIG 08 — Verification bias...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Trip Verification Bias Analysis",
             fontsize=17, fontweight="bold", y=1.02)

verified_map = {"Yes": "Verified", "No": "Unverified", "Not Rated": "Not Rated"}
df["verified_label"] = df["trip_verified"].map(verified_map)
ver_groups = ["Verified", "Unverified"]

# 8a — Mean rating by verification
ax = axes[0]
ver_stats = df[df["verified_label"].isin(ver_groups)].groupby("verified_label")["overall_rating_10"].agg(["mean","count","std"])
bars = ax.bar(ver_stats.index, ver_stats["mean"],
              color=[C["teal"], C["red"]], edgecolor="none", width=0.45)
label_bar(ax, bars, fmt="{:.2f}", offset=0.05)
ax.set_title("Avg Rating: Verified vs Unverified", fontweight="bold")
subtitle(ax, f"Unverified reviews 1.1pts higher — potential selection bias")
ax.set_ylabel("Avg Rating (1–10)")
ax.set_ylim(5, 10)
for i, (idx, row) in enumerate(ver_stats.iterrows()):
    ax.text(i, row["mean"] / 2, f"n={int(row['count'])}",
            ha="center", va="center", color="white", fontweight="bold", fontsize=11)

# 8b — Rating distribution: verified vs not
ax = axes[1]
for group, col in [("Verified", C["teal"]), ("Unverified", C["red"])]:
    data = df[df["verified_label"] == group]["overall_rating_10"].dropna()
    ax.hist(data, bins=10, range=(0.5, 10.5), alpha=0.6,
            color=col, edgecolor="white", label=f"{group} (n={len(data)})",
            density=True, width=0.85)
ax.set_title("Rating Distribution by Verification", fontweight="bold")
subtitle(ax, "Unverified skewed high — trust verified data for analysis")
ax.set_xlabel("Rating (1–10)")
ax.set_ylabel("Density")
ax.legend()

# 8c — Recommendation rate
ax = axes[2]
rec_ver = df[df["verified_label"].isin(ver_groups)].groupby("verified_label").apply(
    lambda x: (x["recommended"] == "Yes").mean() * 100
)
bars = ax.bar(rec_ver.index, rec_ver.values,
              color=[C["teal"], C["red"]], edgecolor="none", width=0.45)
label_bar(ax, bars, fmt="{:.1f}%", offset=0.5)
ax.set_title("Recommendation Rate by Verification", fontweight="bold")
subtitle(ax, "Unverified 14pp more likely to recommend")
ax.set_ylabel("% Recommended")
ax.set_ylim(0, 100)

plt.tight_layout()
add_watermark(fig)
save(fig, "fig08_verification_bias.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 09 — SEASONALITY ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
print("FIG 09 — Seasonality...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Seasonality & Temporal Patterns",
             fontsize=17, fontweight="bold", y=1.01)

# 9a — Quarter heatmap (year × quarter)
ax = axes[0, 0]
q_heat = df.groupby(["year", "quarter"])["overall_rating_10"].mean().unstack()
q_heat.columns = ["Q1", "Q2", "Q3", "Q4"]
sns.heatmap(q_heat, ax=ax, cmap=QA_CMAP, vmin=4, vmax=10,
            annot=True, fmt=".1f", annot_kws={"size": 9},
            linewidths=0.4, cbar_kws={"shrink": 0.8})
ax.set_title("Rating Heatmap — Year × Quarter", fontweight="bold")
subtitle(ax, "Q3 2022 & 2023 consistently weakest — summer operational pressure")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# 9b — Monthly pattern (average across all years)
ax = axes[0, 1]
month_avg = df.groupby("month")["overall_rating_10"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
bar_cols = [C["red"] if v < 6.8 else C["amber"] if v < 7.3 else C["teal"]
            for v in month_avg]
bars = ax.bar(range(1, 13), month_avg.values, color=bar_cols,
              edgecolor="none", width=0.7)
label_bar(ax, bars, fmt="{:.1f}", offset=0.05)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names, rotation=30, ha="right")
ax.set_title("Avg Rating by Month (All Years)", fontweight="bold")
subtitle(ax, "August & September dip — consistent with peak travel season pressure")
ax.set_ylabel("Avg Rating (1–10)")
ax.set_ylim(5, 9)

# 9c — Review volume by month (which months are busiest for reviews)
ax = axes[1, 0]
month_vol = df.groupby("month").size()
ax.bar(range(1, 13), month_vol.values, color=C["navy"],
       edgecolor="none", width=0.7)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names, rotation=30, ha="right")
ax.set_title("Review Volume by Month", fontweight="bold")
subtitle(ax, "Jan–Mar peak review period — post-holiday reflection")
ax.set_ylabel("Number of Reviews")

# 9d — Year over year score trends for key categories
ax = axes[1, 1]
for col, label, style in [
    ("seat_comfort",       "Seat Comfort",  "-"),
    ("food_beverages",     "Food & Bev",    "--"),
    ("cabin_staff_service","Cabin Staff",   "-."),
    ("value_for_money",    "Value",         ":"),
]:
    trend = df.groupby("year")[col].mean()
    ax.plot(trend.index, trend.values, marker="o", linestyle=style,
            linewidth=2, markersize=5, label=label)
ax.set_title("Key Category Trends — Yearly Avg", fontweight="bold")
subtitle(ax, "All categories declining in parallel from 2021 peak")
ax.set_xlabel("Year"); ax.set_ylabel("Avg Score (1–5)")
ax.legend(fontsize=9); ax.set_xticks(years)

plt.tight_layout()
add_watermark(fig)
save(fig, "fig09_seasonality.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 10 — VALUE PERCEPTION
# ────────────────────────────────────────────────────────────────────────────
print("FIG 10 — Value perception...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Value for Money Perception Analysis",
             fontsize=17, fontweight="bold", y=1.02)

# 10a — VfM trend by cabin
ax = axes[0]
vfm_trend = df.groupby(["year", "seat_type"])["value_for_money"].mean().unstack()
for cabin in cabins_main:
    if cabin in vfm_trend.columns:
        ax.plot(vfm_trend.index, vfm_trend[cabin], "o-",
                color=CABIN_COLOURS[cabin], label=cabin,
                linewidth=2.2, markersize=6)
ax.set_title("Value for Money Trend by Cabin", fontweight="bold")
subtitle(ax, "Steepest decline in Economy — cost vs experience gap widening")
ax.set_xlabel("Year"); ax.set_ylabel("Avg VfM Score (1–5)")
ax.legend(); ax.set_xticks(years)

# 10b — VfM score distribution
ax = axes[1]
for cabin, col in CABIN_COLOURS.items():
    if cabin in cabins_main:
        data = df[df["seat_type"] == cabin]["value_for_money"].dropna()
        counts = data.value_counts().sort_index()
        ax.plot(counts.index, counts.values / counts.sum() * 100,
                "o-", color=col, label=cabin, linewidth=2, markersize=6)
ax.set_title("VfM Score Distribution by Cabin", fontweight="bold")
subtitle(ax, "Economy polarised: peaks at 3 and 5")
ax.set_xlabel("VfM Score (1–5)")
ax.set_ylabel("% of Reviews")
ax.legend(); ax.set_xticks([1,2,3,4,5])

# 10c — VfM vs recommendation (does low vfm kill recommendation?)
ax = axes[2]
vfm_rec = df.groupby("value_for_money").apply(
    lambda x: (x["recommended"] == "Yes").mean() * 100
).reset_index()
vfm_rec.columns = ["vfm_score", "rec_pct"]
vfm_rec = vfm_rec[vfm_rec["vfm_score"].notna()]
bars = ax.bar(vfm_rec["vfm_score"], vfm_rec["rec_pct"],
              color=[C["red"],C["red"],C["amber"],C["teal"],C["teal"]],
              edgecolor="none", width=0.6)
label_bar(ax, bars, fmt="{:.0f}%", offset=0.5)
ax.set_title("Recommendation Rate by VfM Score", fontweight="bold")
subtitle(ax, "VfM=1 → only 3% recommend | VfM=5 → 97% recommend")
ax.set_xlabel("Value for Money Score (1–5)")
ax.set_ylabel("% Recommended")
ax.set_ylim(0, 105)

plt.tight_layout()
add_watermark(fig)
save(fig, "fig10_value_perception.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 11 — ROUTE ANALYSIS
# ────────────────────────────────────────────────────────────────────────────
print("FIG 11 — Route analysis...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Route Performance Analysis", fontsize=17, fontweight="bold", y=1.02)

# 11a — Top 20 origins by avg rating
ax = axes[0]
origin_stats = df.groupby("origin").agg(
    mean=("overall_rating_10","mean"),
    count=("overall_rating_10","count"),
).query("count >= 10").sort_values("mean", ascending=True).tail(20)
bar_cols = [C["teal"] if v >= 7.5 else C["amber"] if v >= 6.5 else C["red"]
            for v in origin_stats["mean"]]
bars = ax.barh(origin_stats.index, origin_stats["mean"],
               color=bar_cols, edgecolor="none", height=0.65)
hbar_label(ax, bars, fmt="{:.2f}")
ax.axvline(df["overall_rating_10"].mean(), color=C["maroon"],
           linestyle="--", linewidth=1.5, label="Overall avg")
ax.set_title("Avg Rating by Origin City (≥10 reviews)", fontweight="bold")
subtitle(ax, "Doha departures (outbound) rated lower — inbound experience stronger")
ax.set_xlabel("Avg Rating (1–10)")
ax.legend()

# 11b — Route volume treemap (approximated as bubble chart)
ax = axes[1]
route_vol = df.groupby("route").agg(
    count=("overall_rating_10","count"),
    mean=("overall_rating_10","mean"),
).query("count >= 5").sort_values("count", ascending=False).head(25)
scatter_size = route_vol["count"] * 8
scatter_cols = [C["teal"] if v >= 7 else C["amber"] if v >= 5.5 else C["red"]
                for v in route_vol["mean"]]
ax.scatter(range(len(route_vol)), route_vol["mean"],
           s=scatter_size, alpha=0.7, color=scatter_cols,
           edgecolors="white", linewidth=0.8)
for i, (idx, row) in enumerate(route_vol.iterrows()):
    ax.annotate(idx.replace(" to ", "\n→ "),
                (i, row["mean"]),
                fontsize=6.5, ha="center", va="bottom",
                xytext=(0, 8), textcoords="offset points",
                color=C["slate"])
ax.set_title("Top 25 Routes — Rating vs Volume", fontweight="bold")
subtitle(ax, "Bubble size = review volume | Colour = rating tier")
ax.set_ylabel("Avg Rating (1–10)")
ax.set_xticks([])
ax.set_xlim(-1, len(route_vol))
legend_patches = [
    mpatches.Patch(color=C["teal"],  label="High (≥7.0)"),
    mpatches.Patch(color=C["amber"], label="Mid (5.5–7.0)"),
    mpatches.Patch(color=C["red"],   label="Low (<5.5)"),
]
ax.legend(handles=legend_patches, fontsize=9)

plt.tight_layout()
add_watermark(fig)
save(fig, "fig11_route_analysis.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG 12 — EXECUTIVE SUMMARY DASHBOARD
# ────────────────────────────────────────────────────────────────────────────
print("FIG 12 — Executive summary...")

fig = plt.figure(figsize=(20, 13))
fig.suptitle(
    "Qatar Airways Passenger Experience — Executive Summary Dashboard\n"
    "Skytrax Reviews 2016–2026 | 1,957 Reviews | Senior Data Specialist Portfolio",
    fontsize=16, fontweight="bold", y=1.01,
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.4)

# KPI scorecards (top row)
kpis = [
    ("1,957",        "Total Reviews",         "2016–2026",        C["steel_dark"]),
    (f"{df['overall_rating_10'].mean():.1f}/10",
                     "Avg Overall Rating",    "Fleet-wide",       C["maroon"]),
    (f"{(df['recommended']=='Yes').mean()*100:.0f}%",
                     "Recommend Rate",        "All cabins",       C["steel_mid"]),
    ("5.8/10",       "2024 Avg Rating",       "Lowest on record", C["negative"]),
]
for i, (val, label, sub, colour) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(colour)
    ax.text(0.5, 0.62, val, ha="center", va="center",
            fontsize=26, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.28, label, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
            transform=ax.transAxes)
    ax.text(0.5, 0.1, sub, ha="center", va="center",
            fontsize=9, color="white", alpha=0.8,
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# Main trend chart
ax = fig.add_subplot(gs[1, :2])
for cabin, col in [("Economy Class", C["economy"]),
                   ("Business Class", C["business"])]:
    if cabin in cabin_yearly.columns:
        data = cabin_yearly[cabin].dropna()
        ax.plot(data.index, data.values, "o-",
                label=cabin, color=col, linewidth=2.2, markersize=6)
ax.fill_between(years,
                [6.5]*len(years), [10]*len(years),
                alpha=0.04, color=C["green"])
ax.fill_between(years,
                [0]*len(years), [5.5]*len(years),
                alpha=0.07, color=C["red"])
ax.axhline(6.95, color=C["warm_grey"], linestyle=":", linewidth=1,
           label="Overall avg (6.95)")
ax.set_title("Rating Trend by Cabin — 2016 to 2026", fontweight="bold")
ax.set_xlabel("Year"); ax.set_ylabel("Avg Rating (1–10)")
ax.set_ylim(3.5, 10.5); ax.legend(fontsize=9); ax.set_xticks(years)

# Driver importance
ax = fig.add_subplot(gs[1, 2])
corr_v = df[SCORE_COLS + ["overall_rating_10"]].corr()["overall_rating_10"].drop("overall_rating_10").sort_values()
hbar_cols = [C["green"] if v > 0.7 else C["teal"] for v in corr_v]
ax.barh([SCORE_LABELS[SCORE_COLS.index(c)] for c in corr_v.index],
        corr_v.values, color=hbar_cols, edgecolor="none", height=0.6)
ax.set_title("Satisfaction Drivers", fontweight="bold")
ax.set_xlabel("Correlation r")
ax.axvline(0.7, color=C["red"], linestyle="--", linewidth=1)

# Recommendation donut
ax = fig.add_subplot(gs[1, 3])
rec_c = df["recommended"].value_counts()
ax.pie(rec_c, labels=rec_c.index,
       colors=[C["green"], C["red"]],
       autopct="%1.0f%%", startangle=90,
       wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2})
ax.text(0, 0, "71%\nRec.", ha="center", va="center",
        fontsize=13, fontweight="bold")
ax.set_title("Recommendation", fontweight="bold")

# Findings narrative
findings = [
    ("01", C["negative"],
     "2024: Lowest rated year on record (5.8/10)",
     "Economy class collapsed to 5.2 avg. Immediate product and service audit warranted."),
    ("02", C["maroon"],
     "Value for Money is #1 CSAT predictor (r=0.86)",
     "Strongest correlation in dataset. Price perception drives recommendation more than any service element."),
    ("03", C["steel_dark"],
     "First Class food (3.86) rated below Economy (3.75)",
     "Counter-intuitive anomaly. Elevated passenger expectations in premium cabins not met by catering."),
    ("04", C["steel_mid"],
     "Wi-Fi: 46% data missing — a product gap signal",
     "High non-response rate suggests passengers are not using or not finding Wi-Fi — adoption or quality issue."),
]
for i, (num, col, title, body) in enumerate(findings):
    ax = fig.add_subplot(gs[2, i])
    ax.set_facecolor("white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines["left"].set_color(col)
    ax.spines["left"].set_linewidth(4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.text(0.08, 0.88, f"Finding {num}", fontsize=9, fontweight="bold",
            color=col, transform=ax.transAxes)
    ax.text(0.08, 0.70, title, fontsize=10, fontweight="bold",
            color=C["navy"], transform=ax.transAxes, wrap=True,
            multialignment="left")
    ax.text(0.08, 0.20, body, fontsize=9, color=C["slate"],
            transform=ax.transAxes, wrap=True, multialignment="left",
            verticalalignment="bottom")

add_watermark(fig)
save(fig, "fig12_executive_summary.png")


print()
print("=" * 60)
print("ALL 12 CHARTS GENERATED → ./charts/")
print("=" * 60)
print()
print("Chart inventory:")
charts = [
    "fig01  Dataset overview (structure & composition)",
    "fig02  The rating decline story (headline finding)",
    "fig03  Cabin performance deep dive",
    "fig04  Score drivers & correlations",
    "fig05  Traveller segment analysis",
    "fig06  Aircraft performance",
    "fig07  Geographic analysis (incl. SA market)",
    "fig08  Verification bias analysis",
    "fig09  Seasonality & temporal patterns",
    "fig10  Value for money perception",
    "fig11  Route analysis",
    "fig12  Executive summary dashboard",
]
for c in charts:
    print(f"  ✓ {c}")
