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
│   └── nlp_output_files/
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

```

### Data requirements
Place `qatar_airline_reviews.csv` and `qatar_sentiment_flat.csv` in the working directory before running scripts 01, 03, and 04. The Skytrax review data was scraped from airlinequality.com and is not included in this repository in compliance with the site's terms of use.

---

## Strategic Recommendations for Qatar Airways PDD

The analysis surfaces four actionable product recommendations:

**1. Economy meal service — immediate uplift opportunity**  
A/B test results show a statistically significant +0.30pt satisfaction lift from an elevated economy menu at medium effect size (d=0.609). Complaint rate halved. Economy rollout delivers highest ROI — 70% of passengers, largest absolute impact on NPS.

**2. IFE content refresh cadence — from bi-annual to quarterly**  
IFE scores show a recurring mid-year dip correlating with stale content cycles. Passengers rating IFE ≤2/5 are significantly more likely to be NPS detractors. A quarterly content refresh cycle is a low-cost, high-impact intervention.

**3. Business class catering — close the expectation gap**  
Regression and NLP theme analysis both identify food & beverage as an underperforming dimension relative to Business class passenger expectations. The gap between seat comfort scores (strong) and food scores (weak) in Business class represents the highest-value product investment opportunity in the current portfolio.

**4. Captive advocate programme — address the Advocacy Gap**  
6% of passengers recommend despite net negative sentiment — flying out of necessity not loyalty. These passengers are first to switch when a competitor offers the same route. Identifying and proactively engaging this segment (particularly on high-frequency business routes) is a retention opportunity that standard NPS tracking misses entirely.

---

## Contact

**Edwin Daniels**  
[LinkedIn](www.linkedin.com/in/edwin-daniels) · [Email](edwin.d.daniels@gmail.com) · [Portfolio Microsite]

*This portfolio was developed independently as part of a job application. All analysis and code is original work. The Skytrax review data was collected via public web scraping for non-commercial analytical purposes.*
