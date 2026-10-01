# AI-Powered Collections Strategy: Credit Card Delinquency

Analytics project completed as part of the **Tata iQ GenAI-Powered Data Analytics virtual experience** (individual, simulated consulting engagement). It is a training exercise, not a real client project.

**Scenario:** A lender (Geldium) sees rising credit card delinquency. As an AI transformation consultant, I analyzed customer data, planned an explainable prediction model, wrote a business recommendation, and designed a responsible AI collections system.

## Key findings

- **Data:** 500 customers, 19 columns, 16% delinquent (80 of 500). Missing values in Income (7.8%), Loan_Balance (5.8%) and Credit_Score (0.4%); inconsistent employment labels ("EMP" vs "Employed").
- **Risk signals are weak.** The largest correlation with delinquency is 0.045. Directional patterns only: delinquency rises with credit utilization (12% below 30% to 20% at 70-90%); Business card holders 21.3% vs Platinum 11.8%; Unemployed 19.4% vs Retired 11.5%. None is statistically significant.
- **Data quality issue:** 68 of 77 customers with `Missed_Payments = 0` show at least one "Missed" month in their monthly payment history.
- **Preliminary model check (5-fold CV):** logistic regression AUC 0.45, shallow decision tree AUC 0.54, which is no better than chance. The honest conclusion is to fix the data and validate the model before any deployment.

## Recommendation

An 8-week pilot of proactive reminders plus a flexible payment-plan offer for customers above 50% utilization, with a randomly selected control group. Goal: a 10% relative reduction in 30+ day delinquency versus control (a target to test, not a forecast).

## Project structure

```
code/
  01_eda.py              EDA, cleaning, risk profiling, anomalies, charts
  02_baseline_model.py   Logistic regression vs decision tree (cross-validated)
reports/
  1_EDA_Summary_Report.docx
  2_Model_Plan.docx
  3_Business_Summary_Report.docx
  4_AI_Collections_Strategy_Deck.pdf
images/
  eda_charts.png
data/                    place the dataset here (not included)
```

## How to run

```bash
pip install -r requirements.txt
# place Delinquency_prediction_dataset.xlsx in the data/ folder
python code/01_eda.py
python code/02_baseline_model.py
```

## Approach highlights

- **Cleaning:** standardized labels, median imputation with missing-value flags, no rows dropped.
- **Model choice:** logistic regression for explainability and fairness on a small dataset.
- **Responsible AI:** Age and Location excluded from the model and used only to monitor fairness across groups; supportive interventions only; human approval for sensitive actions.
- **Collections system design:** inputs, decision logic, actions and a learning loop, with autonomous vs human-in-the-loop roles and a phased roadmap.

## Tools

Python, Pandas, Matplotlib, Google Colab, Excel, PowerPoint, Generative AI.

## Limitations

Small, simulated dataset (500 records, 80 delinquent cases). Findings are directional. Fairness checks across groups used small samples and need larger data.

## Disclaimer

The dataset belongs to the program provider and is not redistributed here. Customer IDs are placeholders from a simulated dataset.

## Author

Apoorv Bajpai, MBA (Marketing and Business Analytics)
