"""Exploratory data analysis: data quality, cleaning, risk profiling, anomalies, charts."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
os.makedirs("outputs", exist_ok=True)

df = pd.read_excel("data/Delinquency_prediction_dataset.xlsx")
target = "Delinquent_Account"

# 1. Structure and quality
print("Shape:", df.shape)
miss = df.isnull().sum()
print(pd.DataFrame({"count": miss, "pct": (miss / len(df) * 100).round(1)})[miss > 0])
print("Duplicate rows:", df.duplicated().sum(), "| Duplicate IDs:", df["Customer_ID"].duplicated().sum())
print(df[target].value_counts(normalize=True).round(3))

# 2. Cleaning
df["Employment_Status"] = (df["Employment_Status"].str.strip().str.lower()
                           .replace({"emp": "employed"}).str.title())
for c in ["Income", "Loan_Balance", "Credit_Score"]:
    df[c + "_missing"] = df[c].isnull().astype(int)
    df[c] = df[c].fillna(df[c].median())
month_cols = [c for c in df.columns if c.startswith("Month_")]
df["monthly_missed"] = (df[month_cols] == "Missed").sum(axis=1)
df["monthly_late"] = (df[month_cols] == "Late").sum(axis=1)

# 3. Risk profiling
num_cols = ["Age", "Income", "Credit_Score", "Credit_Utilization", "Debt_to_Income_Ratio",
            "Loan_Balance", "Account_Tenure", "Missed_Payments", "monthly_missed", "monthly_late"]
print("\nMeans by delinquency status:\n", df.groupby(target)[num_cols].mean().round(2).T)
print("\nCorrelation with target:\n", df[num_cols].corrwith(df[target]).round(3).sort_values())
for c in ["Employment_Status", "Credit_Card_Type", "Location"]:
    print(f"\nDelinquency rate by {c}:\n", df.groupby(c)[target].agg(["mean", "count"]).round(3))
df["Util_Band"] = pd.cut(df["Credit_Utilization"], [0, .3, .5, .7, .9, 2])
print("\nBy utilization band:\n", df.groupby("Util_Band", observed=True)[target].agg(["mean", "count"]).round(3))

# 4. Anomalies
zero_mp = df["Missed_Payments"] == 0
print("\nMissed_Payments = 0 but a 'Missed' month:", (zero_mp & (df["monthly_missed"] > 0)).sum(), "of", zero_mp.sum())
print("Delinquent with credit score > 700:", ((df[target] == 1) & (df["Credit_Score"] > 700)).sum())
print("Utilization > 1:", (df["Credit_Utilization"] > 1).sum())

# 5. Charts
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
sns.countplot(x=target, data=df, ax=ax[0, 0]); ax[0, 0].set_title("Class imbalance")
sns.boxplot(x=target, y="Credit_Utilization", data=df, ax=ax[0, 1]); ax[0, 1].set_title("Utilization vs delinquency")
df.groupby("Util_Band", observed=True)[target].mean().plot(kind="bar", ax=ax[1, 0])
ax[1, 0].set_title("Delinquency rate by utilization band")
sns.heatmap(df[num_cols + [target]].corr(), cmap="coolwarm", center=0, ax=ax[1, 1]); ax[1, 1].set_title("Correlation heatmap")
plt.tight_layout(); plt.savefig("outputs/eda_charts.png", dpi=150)

df.drop(columns=["Util_Band"]).to_csv("outputs/cleaned_delinquency_data.csv", index=False)
print("\nSaved outputs/eda_charts.png and outputs/cleaned_delinquency_data.csv")
