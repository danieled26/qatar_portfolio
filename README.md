# Qatar Airways — Passenger Experience Analytics Portfolio
### Edwin Daniels | Senior Data Specialist Application | Product Development & Design Team

---

## Overview

This repository contains a comprehensive data analytics portfolio developed as part of my application for the Senior Data Specialist role within Qatar Airways' Product Development and Design (PDD) team.

The work demonstrates end-to-end analytical capability — from raw data ingestion through to executive-ready insight — using real passenger review data enriched with a modern LLM-powered sentiment pipeline. Every analytical choice was made with the PDD team's core mandate in mind: **translating data into product decisions that improve the passenger experience.**

---

## The Data

### Primary Dataset — Real World
**Source:** Skytrax (airlinequality.com) — scraped directly  
**Coverage:** 1,957 verified passenger reviews | May 2016 – June 2026  
**Fields:** 26 variables including overall rating, category scores (seat comfort, cabin staff, food & beverage, IFE, ground service, Wi-Fi, value for money), cabin class, traveller type, route, aircraft, reviewer country, recommendation status, and full review text

This is real passenger voice data — not simulated, not anonymised, not aggregated. Every insight derived from it reflects genuine passenger sentiment.

### Supplementary Dataset — Synthetic Operational Data
To demonstrate capability across the full analytical scope of the role, five synthetic datasets were generated using statistically realistic distributions:

| Dataset | Rows | Purpose |
|---|---|---|
| Passenger Survey | 48,000 | Satisfaction scores, NPS, monthly KPI tracking |
| Flight Operations | 8,500 | OTD, delays, load factor, fuel efficiency |
| Onboard Product Usage | 35,000 | IFE, Wi-Fi, upgrades, ancillary revenue |
| A/B Test — Meal Service | 6,000 | Experimental meal service trial |
| Customer Profiles | 22,000 | Loyalty segmentation, churn risk, RFM |

> **Note:** Synthetic datasets were designed with realistic distributions, built-in anomalies, and cross-dataset relationships. They are available in the `05_data/synthetic/` directory with full documentation.

---

## Portfolio Pieces

### Piece 01 — Exploratory Data Analysis
**Script:** `01_eda/qatar_eda_2.py`  
**Data:** Real Skytrax reviews  
**Output:** 12 publication-quality charts

A comprehensive EDA leaving no dimension unexplored — temporal trends, cabin performance, satisfaction drivers, aircraft analysis, geographic intelligence, verification bias, seasonality, and value perception. Built in Python using Pandas, Matplotlib, and Seaborn with Qatar Airways corporate brand identity applied throughout.

**Key findings:**
- 2024 is Qatar Airways' worst rated year on record (5.8/10 avg) — a 2.4pt decline from the 2021 peak
- Value for Money is the single strongest predictor of recommendation (Pearson r = 0.86)
- First Class food scores (3.86) sit below Economy (3.75) — a counter-intuitive product anomaly
- Wi-Fi non-response rate of 46% is a structural product gap signal, not missing data
- Verified reviews score 1.1pts lower than unverified — methodological finding with implications for how raw ratings should be interpreted

---

### Piece 02 — NLP Sentiment Analysis Pipeline
**Scripts:** `02_nlp/a_grok_qatar_review_analysis.ipynb` · `02_nlp/b_ollama_qatar_review_analysis.ipynb`  
**Data:** Real Skytrax review text (1,957 reviews)  
**Output:** Structured JSON + 8 Power BI-ready CSVs + word clouds + sentiment trend dashboard

A modern LLM-powered NLP pipeline replacing traditional keyword-matching approaches (VADER, TextBlob) with contextually-aware extraction using Llama 3.1 8B. The pipeline extracts a structured schema per review including sentiment score, primary emotion, sarcasm detection, product theme classifications across 10 dimensions, competitor mentions, rebook intent, and product implications framed in PDD language.

**Pipeline architecture:**
```
Raw review text
    → Groq API / Ollama (Llama 3.1 8B)
    → Structured JSON extraction (40+ fields per review)
    → Flattened CSV for Power BI
    → Keyword splitting for frequency analysis
    → Monthly aggregation for trend analysis
```

**The Passenger Sentiment Index (PSI):**  
Sentiment scores (-1 to 1) were min-max normalised to a 0–100 index anchored to the actual data distribution. This makes the metric immediately interpretable by non-technical stakeholders without arbitrary rescaling assumptions.

**Key methodological choices:**
- Temperature set to 0.1 for consistent structured output
- Incremental saving every 25 reviews — resilient to API disconnects
- Two pipeline variants: Groq API (speed) and Ollama local (no rate limits)
- Sarcasm detection flag — surfaces reviews where emotional tone contradicts the star rating

**The Advocacy Gap:**  
PSI (65) trails Recommendation Rate (71%) by 6 points — indicating a segment of passengers recommending despite net negative sentiment. These captive advocates represent latent churn risk, particularly vulnerable to competitive route entry.

---

### Piece 03 — Semantic Complaint & Praise Clustering
**Script:** `03. nlp_clustering/03_nlp_review_split.ipynb`  
**Data:** NLP pipeline output (top_complaint + top_praise fields)  
**Output:** Clustered CSVs + UMAP visualisations + Power BI frequency tables

Addresses the fundamental problem with raw complaint frequency tables: semantically identical complaints appear as separate entries ("rude staff", "dismissive crew", "unhelpful attendant") understating the true scale of each product issue.

**Method:**
1. Encode each complaint/praise using `all-MiniLM-L6-v2` sentence embeddings (384-dimensional semantic vectors)
2. Normalise embeddings to unit length — clustering by semantic direction, not text length
3. Silhouette analysis to determine optimal cluster count (k=4 to k=11 tested)
4. K-Means clustering on normalised embeddings
5. Auto-label each cluster using term frequency extraction with aviation-specific stopwords
6. UMAP dimensionality reduction to 2D for visualisation

**Why sentence embeddings over TF-IDF or LDA:**  
Traditional approaches cluster on word overlap. Sentence embeddings cluster on *meaning*. "No legroom" and "seat too small" share no words but are semantically identical — embeddings capture this; TF-IDF does not.

**Output for Power BI:**  
Each complaint and praise carries its cluster label, enabling a product manager to filter by cluster ("Seat Comfort Issues") and immediately see which routes, cabins, and time periods drive that cluster — without reading 400 individual complaints.

---

### Piece 04 — Power BI Dashboard
**Data:** Real Skytrax + NLP sentiment + clustering outputs  
**Live dashboard:** [Embedded in portfolio microsite] [View PowerBI Dashboard](https://app.powerbi.com/view?r=eyJrIjoiYmQ1ZjYyMjItNDRkZS00MjJjLTk4MDItYTg2NDhhYmM5OWU3IiwidCI6IjkzYWVkYmRjLWNjNjctNDY1Mi1hYTEyLWQyNTBhODc2YWU3OSJ9&pageName=ReportSectiondfe720a64c6cd2186cbd)

Three-page interactive dashboard built for a PDD product manager audience — not a data scientist audience. Every visual answers a question a product team genuinely has.

**Page 1 — The Big Picture:** Overall rating trend, sentiment index, recommendation rate, year-over-year performance, polarisation analysis

**Page 2 — Passenger Voice:** NLP-powered — product theme heatmap, complaint clusters, praise clusters, emotion distribution, sarcasm detection, competitor mentions, Advocacy Gap metric

**Page 3 — Route & Geographic Intelligence:** Market-level sentiment, route scorecards, aircraft performance, high-volume dissatisfied markets as commercial risk signals

**Key DAX measures:**
```dax
Passenger Sentiment Index = 
VAR MinScore = MINX(ALL(Sentiment), Sentiment[sentiment_score])
VAR MaxScore = MAXX(ALL(Sentiment), Sentiment[sentiment_score])
RETURN DIVIDE(
    Sentiment[sentiment_score] - MinScore,
    MaxScore - MinScore
) * 100

Advocacy Gap = [Recommendation Rate %] - [Sentiment Index Avg]

YoY Rating Change = 
VAR CurrentYear = SELECTEDVALUE(DateTable[Year])
VAR Prior = CALCULATE(
    AVERAGE(Reviews[overall_rating_10]),
    DateTable[Year] = CurrentYear - 1)
RETURN AVERAGE(Reviews[overall_rating_10]) - Prior
```

---

### Piece 05 — A/B Test: Elevated Meal Service
**Script:** `04_ab_test/04_qatar_ab_test.py`  
**Data:** Synthetic meal service trial dataset (n=6,000)  
**Output:** 5 charts + full statistical report

A complete experimental analysis of a hypothetical elevated meal service trial across 4 long-haul routes over 4 months. Demonstrates rigorous experimental methodology including balance verification, parametric and non-parametric testing, effect size calculation, subgroup analysis, and financial business case projection.

**Verdict:** Roll out Treatment menu — high confidence

| Metric | Control | Treatment | Significance |
|---|---|---|---|
| Meal satisfaction | 3.607 | 3.906 | p<0.001 *** |
| Complaint rate | 8.0% | 4.3% | p<0.001 *** |
| Finish rate | 77.7% | 87.6% | p<0.001 *** |
| Effect size | — | — | Cohen's d=0.609 (Medium) |

**Subgroup finding:** Economy shows largest lift (d=0.850, Large effect). First Class lift negligible — already at satisfaction ceiling. Economy rollout delivers highest ROI per dollar of meal cost uplift.

**Statistical tests applied:**
- Two-sample t-test (primary)
- Mann-Whitney U (non-parametric confirmation)
- Chi-squared test (complaint rate)
- Shapiro-Wilk (normality assumption check)
- Cohen's d effect size
- 95% confidence intervals on lift estimate

---

### Piece 06 — Recommendation Propensity Model
**Script:** `05_regression/05_propensity_model.py`  
**Data:** Real Skytrax reviews + NLP sentiment pipeline output (joined on review_id)  
**Output:** 9 charts + feature importance CSV + propensity results CSV + trained model (`.pkl`)

A classification model predicting whether a passenger will recommend Qatar Airways (Yes/No), used as a loyalty/churn proxy. True churn is unobservable in a reviews dataset — recommendation intent is the closest available signal. Reichheld's NPS research validates this: a passenger who says "I would not recommend" has mentally defected regardless of whether they fly again on price or route convenience.

The model's primary output is not the accuracy score but the **product investment prioritisation framework** — a ranked table of which experience dimensions most strongly drive recommendation intent, with direction of effect and plain-English product implications for each.

**Three models, in sequence:**
1. **Logistic Regression** — baseline + interpretable coefficient signs (what pushes toward recommend vs not)
2. **Random Forest** — non-linear importance + Partial Dependence Plots (relationship shape, not just rank)
3. **XGBoost** — performance benchmark + SHAP decomposition (per-feature, per-passenger contribution)

**Model performance:**

| Model | 5-Fold CV AUC | Test AUC | Test Accuracy | F1 (Not Recommend) | F1 (Recommend) |
|---|---|---|---|---|---|
| Logistic Regression | 0.9892 ± 0.0045 | **0.9879** | 96% | 0.93 | 0.97 |
| Random Forest | 0.9877 ± 0.0051 | 0.9850 | 94% | 0.89 | 0.95 |
| XGBoost | 0.9881 ± 0.0045 | 0.9837 | 95% | 0.91 | 0.96 |

AUC scores above 0.98 across all models reflect the strong internal consistency of passenger review data — passengers who rate dimensions poorly also withhold recommendation. Logistic Regression marginally outperforms the tree models on test AUC, confirming the relationship is largely linear at portfolio level.

**Top 10 features by XGBoost importance (product & NLP features only):**

| Rank | Feature | XGB Importance | Direction |
|---|---|---|---|
| 1 | Overall Rating (1–10) | 0.3664 | ↑ Recommend |
| 2 | Would Rebook Signal | 0.2428 | ↑ Recommend |
| 3 | NLP Theme: Value | 0.0603 | ↑ Recommend |
| 4 | NLP Sentiment Score | 0.0189 | ↑ Recommend |
| 5 | Value for Money Score | 0.0182 | ↑ Recommend |
| 6 | Ground Service Score | 0.0161 | ↑ Recommend |
| 7 | NLP Theme: Cabin Crew | 0.0150 | ↑ Recommend |
| 8 | IFE Score | 0.0141 | ↑ Recommend |
| 9 | NLP Theme: IFE | 0.0138 | ↑ Recommend |
| 10 | Cabin Staff Score | 0.0134 | ↑ Recommend |

> **Note on Overall Rating and Would Rebook Signal:** Both rank at the top because they are near-outcome proxies — overall rating aggregates all product dimensions, and rebook intent is behaviourally adjacent to recommendation. The actionable investment framework sits in ranks 3–10: the specific product and NLP theme features that independently drive recommendation above and beyond the passenger's general sentiment.

**Key modelling decisions:**
- Target: `recommended` (Yes=1, No=0) — 71.2% positive, 2.48:1 imbalance → `class_weight='balanced'` applied
- Train/test split: 80/20, stratified on target
- Missing numeric values imputed with median; categorical NaN filled before one-hot encoding
- Numeric features scaled (StandardScaler) for Logistic Regression only
- Class imbalance handled via `class_weight='balanced'` (LR, RF) and `scale_pos_weight` (XGBoost)
- 5-fold stratified cross-validation on training set before final test evaluation

---

## Technical Stack

| Category | Tools |
|---|---|
| **Languages** | Python 3.12 |
| **Data manipulation** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn, WordCloud |
| **Statistics** | SciPy, Scikit-learn |
| **NLP / LLM** | Groq API, Ollama, Llama 3.1 8B |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Dimensionality reduction** | UMAP |
| **Propensity Modelling** | XGBoost, SHAP |
| **BI / Dashboarding** | Microsoft Power BI |
| **Deployment** | Vercel (microsite), GitHub Pages |
| **Version control** | Git / GitHub |

---

## Repository Structure

```
qatar-airways-pdd-portfolio/
│
├── README.md
│
├── 00_scraper/
│   └── 01_qatar_airline_reviews_scraper.py
│   └── 02_qatar_seat_reviews_scraper.py
│   └── 03_qatar_lounge_reviews_scraper.py
│   └── 04_scraper.ipynb
│   └── 05_qatar_tripadvisor_reviews_scraper #does not work - gets blocked 
│
├── 01_eda/
│   └── qatar_eda.py                  # 12-chart EDA pipeline
│   └── qatar_eda_2.py                # Corrected with Qatar colours
│
├── 02_nlp/
│   ├── a_grok_qatar_review_analysis.ipynb   # Groq API required
│   ├── b_ollama_qatar_review_analysis.ipynb   # Ollama local model pipeline
│
├── 03. nlp_clustering/
│   └── 03_nlp_review_split.ipynb     # Keyword split and Semantic complaint/praise clustering
│
├── 04_ab_test/
│   └── 04_qatar_ab_test.py           # Full A/B test analysis
│   └── 04_notebook_ab_tests.ipynb    # Colab notebook
│
├── 05_regression/
│   └── 05_propensity_model.py        # Recommendation propensity model (LR + RF + XGBoost + SHAP)
│
├── 05_data/
│   ├── scraped/
│       ├── qatar_airline_reviews.csv
│       ├── qatar_lounge_reviews.csv
│       ├── qatar_seat_reviews.csv                     
│   └── synthetic/
│       ├── 01_passenger_survey.csv
│       ├── 02_flight_operations.csv
│       ├── 03_product_usage.csv
│       ├── 04_ab_test_meal.csv
│       └── 05_customer_profiles.csv
│
└── 06_outputs/
│   └── charts/
│       ├── 01_eda_charts/
│       ├── 02_nlp_charts/
│       ├── 03_ab_test_charts/
│       ├── 05_regression_charts/
│   └── nlp_output_files/
│   └── propensity_model/             # Model pkl + results CSV + feature importance CSV
└── 07_branding_elements/ # used for power bi dashboard and slides

```

---

## Running the Code

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn \
            sentence-transformers umap-learn wordcloud groq ollama
```

### Execution order
```bash
# 0. Scrape Data
python 00_scraper/01_qatar_airline_reviews_scraper.py

# 1. EDA
python 01_eda/qatar_eda_2.py

# 2a. NLP (Groq API — requires GROQ_API_KEY)
VSCode/Colab 02_nlp/a_grok_qatar_review_analysis.ipynb #run on colab for compute resources
                OR 
# 2b. NLP (Ollama local — requires ollama serve + model pulled)
VSCode/Colab 02_nlp/b_ollama_qatar_review_analysis.ipynb

# 3. Keyword splitting and Semantic Clustering for Power BI
VSCode/Colab 03. nlp_clustering/03_nlp_review_split.ipynb

# 4. A/B test analysis
VSCode/Colab 04_ab_test/04_notebook_ab_tests.ipynb #Run on colab
                        OR
python 04_ab_test/04_qatar_ab_test.py

# 5. Recommendation propensity model (requires NLP output from step 2)
pip install xgboost shap
python 05_regression/05_propensity_model.py

```

### Data requirements
Place `qatar_airline_reviews.csv` and `qatar_sentiment_flat.csv` in the working directory before running scripts 01, 03, and 04. The Skytrax review data was scraped from airlinequality.com and is not included in this repository in compliance with the site's terms of use.

---

## Strategic Recommendations for Qatar Airways PDD

The analysis surfaces five actionable product recommendations:

**1. Economy meal service — immediate uplift opportunity**  
A/B test results show a statistically significant +0.30pt satisfaction lift from an elevated economy menu at medium effect size (d=0.609). Complaint rate halved. Economy rollout delivers highest ROI — 70% of passengers, largest absolute impact on NPS.

**2. IFE content refresh cadence — from bi-annual to quarterly**  
IFE scores show a recurring mid-year dip correlating with stale content cycles. Passengers rating IFE ≤2/5 are significantly more likely to be NPS detractors. A quarterly content refresh cycle is a low-cost, high-impact intervention.

**3. Value for money perception — the primary investment lever**  
The propensity model identifies Value for Money as the single strongest product-level predictor of recommendation intent (top 3 features after removing outcome proxies overall rating and rebook signal). Both the numeric score (rank 5) and the NLP free-text theme (rank 3) surface independently — confirming this is not a scoring artefact but a genuine passenger perception gap. Any product investment must be accompanied by a narrative that makes the quality improvement legible to passengers at point of purchase and onboard. EDA separately confirms that Business class food scores (avg 3.86) sit below seat comfort — a specific product mismatch within an otherwise premium cabin that contributes to this value perception gap.

**4. Ground service — an underweighted investment area**  
The propensity model ranks Ground Service above Food & Beverage, Seat Comfort, and Wi-Fi (rank 6 by XGBoost importance). Airport touchpoints set expectations before boarding; poor ground service creates a negative prior that degrades onboard experience ratings. High-volume origin markets with low ground service scores represent the highest-concentration churn risk in the current portfolio.

**5. Captive advocate programme — address the Advocacy Gap**  
6% of passengers recommend despite net negative sentiment — flying out of necessity not loyalty. These passengers are first to switch when a competitor offers the same route. The propensity model's Would Rebook Signal (rank 2) directly identifies this segment: passengers who signal "maybe" represent latent churn invisible to standard NPS tracking. Proactive engagement on high-frequency business routes is the highest-value retention intervention.

---

## Contact

**Edwin Daniels**  
[LinkedIn](https://www.linkedin.com/in/edwin-daniels) · [Email](malito:edwin.d.daniels@gmail.com) · [Portfolio Microsite]

*This portfolio was developed independently as part of a job application. All analysis and code is original work. The Skytrax review data was collected via public web scraping for non-commercial analytical purposes.*
