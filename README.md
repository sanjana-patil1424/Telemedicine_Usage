# Telemedicine_Usage
Data science case study on telemedicine usage and access gaps using a reproducible Python-simulated survey dataset (6,000 cleaned respondents, modeled on HINTS 6) covering data cleaning, variable classification, exploratory analysis, and chi-square testing.
# Telemedicine Usage and Access: A Data Science Case Study

A **Fundamentals of Data Science** (Module 1) case study examining patterns and access barriers in adult telemedicine (care via phone or video)[cite: 1]. Using a reproducible Python-simulated cross-sectional survey dataset modeled on the US NCI **Health Information National Trends Survey 6 (HINTS 6)**, this project covers problem framing, data type and scale classification, data cleaning, exploratory visualisations, and statistical association checks[cite: 1, 2, 5, 6].

> **Disclaimer:** Because this dataset is synthetically generated in Python (`NumPy`, seed `42`) to demonstrate data science workflows, the numbers and subgroup differences describe patterns in this simulated dataset only and do not serve as real-world epidemiological proof[cite: 2, 9, 10].

---

## Problem Statement & Objectives

- **Problem Statement:** Using a simulated survey dataset modeled on an open national health survey, identify patterns in telehealth adoption among adults and determine which groups experience access gaps[cite: 1, 2].
- **Problem Type:** Descriptive and exploratory (what is happening, and for whom)[cite: 1].
- **Objectives:**
  1. Define the telehealth access problem in measurable terms[cite: 1].
  2. Identify structured, semi-structured, and unstructured fields/concepts[cite: 1, 2].
  3. Classify variables as numerical or categorical and assign measurement scales[cite: 1, 3].
  4. Describe survey-style data collection and its analytical caveats[cite: 1, 2].
  5. Detect and handle missing values, duplicate records, and inconsistent text responsibly[cite: 1, 4].
  6. Report usage patterns across demographic and connectivity subgroups, including combined barriers and chi-square significance tests[cite: 1, 6, 9].

---

## Dataset Overview

- **Dataset Name:** Telemedicine Usage and Access Survey (Simulated)[cite: 2]
- **Population Modelled:** Adults aged 18–90 in a cross-sectional questionnaire format (one row per respondent)[cite: 2].
- **Dataset Size:** `6,060` raw rows × `12` columns (comprising `6,000` simulated respondents plus `60` injected duplicate rows, missing values across `6` columns, and inconsistent casing/whitespace in `"male"` and `"urban"`), reduced to `6,000` clean rows after deduplication[cite: 2, 4].
- **Outcome Variable:** `Telehealth_Use` — whether the respondent received care by phone or video in the past 12 months (`Yes` / `No`)[cite: 2].

### Data Types in Context

| Data Type | Meaning | Role in This Case Study |
| :--- | :--- | :--- |
| **Structured** | Fixed rows and columns with defined types[cite: 2] | The 12-column survey CSV (`Age`, `Education`, `Telehealth_Use`, etc.)[cite: 2, 3] |
| **Semi-structured** | Self-describing, tagged, or nested data (JSON, XML)[cite: 2] | Codebook / data dictionary describing fields and codes; JSON from a survey API[cite: 2] |
| **Unstructured** | No fixed format[cite: 2] | Free-text `"Other, please specify"` answers or patient feedback (present in real surveys)[cite: 2] |

### Variable Classification & Measurement Scales (12 Columns)

| Column | Variable Type | Measurement Scale | Values / Notes |
| :--- | :--- | :--- | :--- |
| `Respondent_ID` | Categorical (Identifier)[cite: 3] | Nominal[cite: 3] | `10001` to `16000`[cite: 3] |
| `Age` | Numerical (Discrete)[cite: 3] | Ratio[cite: 3] | `18` to `90` years[cite: 3] |
| `Gender` | Categorical[cite: 3] | Nominal[cite: 3] | `Male`, `Female`[cite: 3] |
| `Education` | Categorical[cite: 3] | Ordinal[cite: 3] | `Below High School` < `High School` < `Some College` < `Graduate`[cite: 3] |
| `Income` | Categorical[cite: 3] | Ordinal[cite: 3] | `<25K` < `25K-50K` < `50K-100K` < `>100K`[cite: 3] |
| `Area` | Categorical[cite: 3] | Nominal[cite: 3] | `Urban`, `Rural`[cite: 3] |
| `Internet_Access` | Categorical (Binary)[cite: 3] | Nominal[cite: 3] | `Yes`, `No`[cite: 3] |
| `General_Health` | Categorical[cite: 3] | Ordinal[cite: 3] | `Poor` < `Fair` < `Good` < `Very Good` < `Excellent`[cite: 3] |
| `Doctor_Visits_12M` | Numerical (Discrete)[cite: 3] | Ratio[cite: 3] | `0` to `12` visits (observed)[cite: 3] |
| `Telehealth_Use` | Categorical (Binary)[cite: 3] | Nominal[cite: 3] | `Yes`, `No`[cite: 3] |
| `Visit_Reason` | Categorical[cite: 3] | Nominal[cite: 3] | `Follow-up`, `New problem`, `Prescription refill`, `Mental health`, `Other` (`Not Applicable` for non-users)[cite: 3] |
| `Satisfaction` | Categorical[cite: 3] | Ordinal (Likert)[cite: 3] | `Very Dissatisfied` < ... < `Very Satisfied` (`Not Applicable` for non-users)[cite: 3] |

---

## Data Preparation & Quality Assessment

- **Completeness:** Before cleaning, `1,562` of `72,720` cells (`2.15%`) were missing across 6 columns (`Income`: `493` / `8.14%`; `General_Health`: `285` / `4.70%`; `Education`: `265` / `4.37%`; `Doctor_Visits_12M`: `203` / `3.35%`; `Internet_Access`: `202` / `3.33%`; `Age`: `114` / `1.88%`)[cite: 4]. Numerical columns (`Age`, `Doctor_Visits_12M`) were filled with the **median**, while categorical columns were assigned an `"Unknown"` category[cite: 4].
- **Skip Logic vs. Missing Values:** `"Not Applicable"` entries in `Visit_Reason` and `Satisfaction` for the `51.4%` non-user group reflect questionnaire skip logic rather than missing values, so they were analyzed among users only without imputation[cite: 5].
- **Uniqueness:** `60` exact duplicate rows (`0.99%` of `6,060`) were identified and removed via `drop_duplicates()`, leaving `6,000` rows[cite: 4].
- **Consistency & Validity:** Standardized casing and whitespace using `str.strip().str.title()` (fixing `"male"` and `"urban"` entries)[cite: 4, 6], checked valid value ranges[cite: 5], and created derived columns `Age_Group` (`18-49`, `50-64`, `65+`) and `Used` (`1` = `Yes`, `0` = `No`)[cite: 4, 6].

---

## Key Findings (Simulated Dataset, Seed `42`)

- **Overall Adoption:** `48.6%` of respondents in the cleaned dataset used telehealth[cite: 7, 9].
- **Age Summary (Cleaned):** Mean `48.08` years, SD `16.14`, range `18` to `90`, median `48`, IQR (`25%`–`75%`) `37` to `59`[cite: 7].
- **Subgroup Usage (% Used Telehealth):**
  - **Education:** `Below High School` (`36.1%`), `High School` (`44.1%`), `Some College` (`48.0%`), `Graduate` (`56.9%`), `Unknown` (`50.8%`) — a `20.8` percentage-point gap from lowest to highest[cite: 7, 9].
  - **Internet Access:** `No` (`36.2%`) vs. `Yes` (`49.9%`) — a `13.7` percentage-point gap (`Unknown`: `48.5%`)[cite: 7, 9].
  - **Income:** `<25K` (`40.6%`), `25K-50K` (`46.2%`), `50K-100K` (`51.2%`), `>100K` (`53.9%`), `Unknown` (`50.3%`) — a `13.3` percentage-point gap[cite: 7, 9].
  - **Area:** `Rural` (`41.9%`) vs. `Urban` (`50.3%`) — an `8.4` percentage-point gap[cite: 7, 9].
  - **Gender:** `Female` (`51.5%`) vs. `Male` (`45.6%`)[cite: 7, 9].
  - **Age Group:** `18-49` (`48.9%`), `50-64` (`50.1%`), `65+` (`44.8%`)[cite: 7, 9].
- **Compounding Barriers & Chi-Square Tests:** Among respondents with known internet status, telehealth usage spans from `51.1%` (`Urban`, internet, `n = 4,331`) down to `31.4%` (`Rural`, no internet, `n = 226`)[cite: 9]. Chi-square tests show `p < 0.001` for education, internet access, area, and gender, and `p = 0.03` for age group[cite: 9].
- **Visit Reasons & User Satisfaction (Users Only):** Top reasons are `Follow-up` (`35.3%`), `New problem` (`24.8%`), `Prescription refill` (`15.7%`), `Mental health` (`13.8%`), and `Other` (`10.5%`)[cite: 7]. Among users, `76.3%` are `Satisfied` (`40.9%`) or `Very Satisfied` (`35.4%`), while `9.9%` are `Dissatisfied` (`7.1%`) or `Very Dissatisfied` (`2.8%`)[cite: 7, 10].

---

## Limitations, Ethics & Future Work

- **Limitations:** Simulated circularity (outcomes generated via synthetic probabilities), non-calibration to real prevalence (HINTS 6 reports roughly `39%` telehealth use)[cite: 2, 10], unweighted sample percentages, random missingness injection, and observational cross-sectional design (association, not causation)[cite: 10].
- **Ethical & Privacy Considerations:** Real health data requires de-identification, small-cell suppression/merging to protect privacy, clear consent and purpose alignment, transparent documentation, and fair framing that highlights infrastructure/support barriers rather than blaming individuals[cite: 10, 11].
- **Future Work:** Replace simulated data with real **HINTS 6** data, fit multivariable models such as logistic regression to isolate predictor effects, and apply text analysis to open-ended responses.

---

## Repository Structure & How to Run

```text
├── telemedicine_dataset.csv     # Raw simulated dataset (6,060 rows x 12 columns)[cite: 2, 6, 12]
├── telemedicine_cleaned.csv     # Cleaned dataset (6,000 rows + derived columns)[cite: 4, 6]
├── fig1_missing.png             # Missing-value heatmap before cleaning[cite: 6, 8]
├── fig2_analysis.png            # Exploratory multi-panel visualisations[cite: 6, 8]
└── analysis.py                  # Full simulation, cleaning, visualisation, and chi-square script[cite: 5, 6]
