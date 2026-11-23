import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# ========== 1. SIMULATE DATASET (6,000 rows, 12 columns) ==========
rng = np.random.default_rng(42)  #[cite: 2, 5]
n = 6000  #[cite: 2, 5]

age = np.clip(rng.normal(48, 17, n).round(), 18, 90).astype(int)  #[cite: 5]
gender = rng.choice(["Male", "Female"], n, p=[0.49, 0.51])  #[cite: 5]
education = rng.choice(
    ["Below High School", "High School", "Some College", "Graduate"],
    n,
    p=[0.10, 0.27, 0.33, 0.30],
)  #[cite: 5]
income = rng.choice(
    ["<25K", "25K-50K", "50K-100K", ">100K"],
    n,
    p=[0.20, 0.25, 0.32, 0.23],
)  #[cite: 5]
area = rng.choice(["Urban", "Rural"], n, p=[0.80, 0.20])  #[cite: 5]
internet = np.where(
    rng.random(n) < np.where(area == "Rural", 0.82, 0.93),
    "Yes",
    "No",
)  #[cite: 5]
health = rng.choice(
    ["Poor", "Fair", "Good", "Very Good", "Excellent"],
    n,
    p=[0.05, 0.15, 0.35, 0.30, 0.15],
)  #[cite: 5]
visits = rng.poisson(np.where(age > 50, 4, 2.5)).clip(0, 20)  #[cite: 5]

edu_s = pd.Series(education).map(
    {"Below High School": -0.5, "High School": -0.2, "Some College": 0.1, "Graduate": 0.4}
).values  #[cite: 5]
inc_s = pd.Series(income).map(
    {"<25K": -0.3, "25K-50K": -0.1, "50K-100K": 0.1, ">100K": 0.3}
).values  #[cite: 5]

z = (
    -0.55
    + edu_s
    + inc_s
    + 0.6 * (internet == "Yes")
    - 0.35 * (area == "Rural")
    + 0.015 * (visits - 3)
    - 0.012 * np.abs(age - 45)
    + 0.2 * (gender == "Female")
)  #[cite: 5]
used = np.where(rng.random(n) < 1 / (1 + np.exp(-z)), "Yes", "No")  #[cite: 5]

satisfaction = np.where(
    used == "Yes",
    rng.choice(
        ["Very Dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very Satisfied"],
        n,
        p=[0.03, 0.07, 0.15, 0.40, 0.35],
    ),
    "Not Applicable",
)  #[cite: 5]
reason = np.where(
    used == "Yes",
    rng.choice(
        ["Follow-up", "New problem", "Prescription refill", "Mental health", "Other"],
        n,
        p=[0.35, 0.25, 0.15, 0.15, 0.10],
    ),
    "Not Applicable",
)  #[cite: 5]

df = pd.DataFrame(
    {
        "Respondent_ID": np.arange(10001, 10001 + n),
        "Age": age,
        "Gender": gender,
        "Education": education,
        "Income": income,
        "Area": area,
        "Internet_Access": internet,
        "General_Health": health,
        "Doctor_Visits_12M": visits,
        "Telehealth_Use": used,
        "Visit_Reason": reason,
        "Satisfaction": satisfaction,
    }
)  #[cite: 5]

# Inject realistic data-quality problems
for col, p in [
    ("Income", 0.08),
    ("Education", 0.04),
    ("Internet_Access", 0.03),
    ("General_Health", 0.05),
    ("Age", 0.02),
    ("Doctor_Visits_12M", 0.03),
]:
    df.loc[rng.random(n) < p, col] = np.nan  #[cite: 6]

df.loc[rng.random(n) < 0.01, "Gender"] = "male"  #[cite: 6]
df.loc[rng.random(n) < 0.01, "Area"] = "urban"  #[cite: 6]
df = pd.concat([df, df.sample(60, random_state=1)], ignore_index=True)  # 60 duplicates[cite: 6]
df.to_csv("telemedicine_dataset.csv", index=False)  #[cite: 6]

# ========== 2. INSPECT ==========
print("Shape:", df.shape)
print(df.dtypes)
print(df.head())  #[cite: 6]

miss = df.isnull().sum().to_frame("missing")
miss["percent"] = (miss["missing"] / len(df) * 100).round(2)
print(miss[miss["missing"] > 0])
print("Duplicate rows:", df.duplicated().sum())  #[cite: 6]

# ========== 3. CLEAN ==========
df = df.drop_duplicates()  #[cite: 6]
df["Gender"] = df["Gender"].str.strip().str.title()  #[cite: 6]
df["Area"] = df["Area"].str.strip().str.title()  #[cite: 6]
df["Age"] = df["Age"].fillna(df["Age"].median())  #[cite: 6]
df["Doctor_Visits_12M"] = df["Doctor_Visits_12M"].fillna(df["Doctor_Visits_12M"].median())  #[cite: 6]

for c in ["Income", "Education", "Internet_Access", "General_Health"]:
    df[c] = df[c].fillna("Unknown")  #[cite: 6]

print("After cleaning:", df.shape, "| missing left:", int(df.isnull().sum().sum()))  #[cite: 6]

df["Age_Group"] = pd.cut(df["Age"], [17, 49, 64, 120], labels=["18-49", "50-64", "65+"])  #[cite: 6]
df["Used"] = (df["Telehealth_Use"] == "Yes").astype(int)  #[cite: 6]

# ========== 4. DESCRIPTIVE ANALYSIS ==========
print(df["Age"].describe().round(2))  #[cite: 6]
print("Overall telehealth use %:", round(df["Used"].mean() * 100, 1))  #[cite: 6]

for col in ["Age_Group", "Education", "Income", "Area", "Internet_Access", "Gender"]:
    print(df.groupby(col, observed=True)["Used"].mean().mul(100).round(1).to_string(), "\n")  #[cite: 6]

print(df.loc[df["Used"] == 1, "Visit_Reason"].value_counts(normalize=True).mul(100).round(1))  #[cite: 6]
print(df.loc[df["Used"] == 1, "Satisfaction"].value_counts(normalize=True).mul(100).round(1))  #[cite: 6]

# ========== 5. VISUALISATION ==========
sns.set_style("whitegrid")  #[cite: 6]
plt.figure(figsize=(8, 4))
sns.heatmap(
    pd.read_csv("telemedicine_dataset.csv").isnull(),
    cbar=False,
    yticklabels=False,
)
plt.title("Missing Value Map (before cleaning)")
plt.tight_layout()
plt.savefig("fig1_missing.png", dpi=120)
plt.show()  #[cite: 6]

fig, ax = plt.subplots(2, 3, figsize=(15, 8))  #[cite: 6]
sns.histplot(df["Age"], bins=20, ax=ax[0, 0], color="steelblue")
ax[0, 0].set_title("Age Distribution")  #[cite: 6]

df.groupby("Age_Group", observed=True)["Used"].mean().mul(100).plot(
    kind="bar", ax=ax[0, 1], color="teal", rot=0
)
ax[0, 1].set_title("Use by Age Group (%)")  #[cite: 6]

order = ["Below High School", "High School", "Some College", "Graduate", "Unknown"]  #[cite: 6]
df.groupby("Education")["Used"].mean().mul(100).reindex(order).plot(
    kind="bar", ax=ax[0, 2], color="orange", rot=20
)
ax[0, 2].set_title("Use by Education (%)")  #[cite: 6]

df.groupby("Area")["Used"].mean().mul(100).plot(
    kind="bar", ax=ax[1, 0], color="green", rot=0
)
ax[1, 0].set_title("Use by Area (%)")  #[cite: 6]

df.groupby("Internet_Access")["Used"].mean().mul(100).plot(
    kind="bar", ax=ax[1, 1], color="purple", rot=0
)
ax[1, 1].set_title("Use by Internet Access (%)")  #[cite: 6]

df.loc[df["Used"] == 1, "Visit_Reason"].value_counts().plot(
    kind="bar", ax=ax[1, 2], color="crimson", rot=20
)
ax[1, 2].set_title("Reason for Telehealth Visit")  #[cite: 6]

for a in ax.flat:
    a.set_xlabel("")  #[cite: 6]

plt.tight_layout()
plt.savefig("fig2_analysis.png", dpi=120)
plt.show()  #[cite: 6]

df.to_csv("telemedicine_cleaned.csv", index=False)  #[cite: 6]

# ========== 6. COMBINED BARRIERS & CHI-SQUARE TESTS ==========
known = df[df["Internet_Access"] != "Unknown"]  #[cite: 6]
print(
    known.pivot_table(
        index="Area",
        columns="Internet_Access",
        values="Used",
        aggfunc=["mean", "count"],
    ).round(3)
)  #[cite: 6]

for c in ["Education", "Internet_Access", "Area", "Gender", "Age_Group"]:
    chi2, p, dof, _ = chi2_contingency(pd.crosstab(df[c], df["Used"]))
    print(f"{c}: chi2={chi2:.1f}, dof={dof}, p={p:.2e}")  #[cite: 6, 7]