"""Preliminary baseline check: logistic regression vs shallow decision tree (5-fold CV).
Run 01_eda.py first. Age and Location are intentionally NOT used as features."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

df = pd.read_csv("outputs/cleaned_delinquency_data.csv")
y = df["Delinquent_Account"]
num = ["Credit_Utilization", "Missed_Payments", "monthly_missed", "monthly_late", "Debt_to_Income_Ratio",
       "Income", "Loan_Balance", "Credit_Score", "Account_Tenure"]
cat = ["Employment_Status", "Credit_Card_Type"]
X = df[num + cat]
pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(drop="first"), cat)])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
models = {
    "Logistic regression": make_pipeline(pre, LogisticRegression(class_weight="balanced", max_iter=1000)),
    "Decision tree (depth 3)": make_pipeline(pre, DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)),
}
for name, m in models.items():
    prob = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (prob >= 0.5).astype(int)
    print(f"{name}: AUC={roc_auc_score(y, prob):.2f} recall={recall_score(y, pred):.2f} "
          f"precision={precision_score(y, pred):.2f} F1={f1_score(y, pred):.2f}")
print("AUC near 0.5 means no better than chance.")
