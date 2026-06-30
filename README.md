# Customer Churn Prediction

> A machine learning project that predicts whether a telecom customer is likely
> to churn, with explainable predictions powered by SHAP. Built end-to-end from
> raw data through model evaluation, using the IBM expanded Telco dataset.

![Top churn drivers](outputs/figures/shap_summary.png)

---

## 📌 Why this project

Customer churn costs telecom companies an estimated 15–25% of annual revenue,
and acquiring a new customer is roughly 5x more expensive than retaining an
existing one. A good churn model lets a business focus retention efforts on
the customers most likely to leave — and explainable predictions help the
business understand *why* a customer is at risk, not just that they are.

I built this project to learn the end-to-end machine learning workflow:
exploring real data, identifying and handling data leakage, engineering
features, comparing models, evaluating honestly, and explaining predictions
in a way a non-technical stakeholder can act on.

---

## 📊 Dataset

**[IBM Telco Customer Churn (expanded)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)**
— 7,043 customers across **33 columns**. This is the expanded IBM version of
the well-known Telco Churn dataset, which adds geographic data, Customer
Lifetime Value (CLTV), and churn reasons on top of the standard demographics,
account info, and services.

- **Target:** `Churn Value` (0 = stayed, 1 = churned)
- **Class balance:** ~27% churners (imbalanced)
- **Notable extras vs. the standard version:** Latitude/Longitude/City for
  geographic analysis, CLTV per customer, and a free-text `Churn Reason` for
  customers who left

### Handling data leakage

Three columns in this dataset are **data leakage** — information that wouldn't
exist at real prediction time:

- `Churn Score` — IBM's own pre-computed risk score (using it would mean
  letting their model do my job)
- `Churn Reason` — only filled in *because* the customer churned
- `Churn Label` — a duplicate of the target

All three are dropped before training. `Churn Reason` is set aside separately
for a standalone analysis (see Notebook 08).

---

## 🛠️ Tech stack

**Language:** Python 3.12
**Data & ML:** pandas, numpy, scikit-learn, XGBoost, SHAP
**Visualization:** matplotlib, seaborn
**Environment:** Jupyter Notebooks in VS Code
**Version control:** Git + GitHub

---

## 📁 Project structure

```
customer-churn-predictor/
├── data/
│   ├── raw/              # original IBM Telco CSV (33 columns)
│   └── processed/        # cleaned data + churn reasons (set aside)
├── notebooks/
│   ├── 01_explore.ipynb        # first look, identify leakage + useless cols
│   ├── 02_clean.ipynb          # drop leakage, fix types, save clean data
│   ├── 03_eda.ipynb            # charts, hypotheses, geographic map
│   ├── 04_preprocess.ipynb     # encode, scale, train/test split
│   ├── 05_train.ipynb          # train + compare 3 models
│   ├── 06_evaluate.ipynb       # test metrics, confusion matrix, SHAP
│   ├── 07_predict.ipynb        # single-customer prediction with explanation
│   └── 08_churn_reasons.ipynb  # bonus: why customers leave (qualitative)
├── outputs/
│   ├── figures/          # exported charts
│   └── models/           # trained model (.pkl)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Approach

### 1. Exploration — `01_explore.ipynb`
First look at shape, types, missing values, and target distribution.
- [ ] Key finding 1 (e.g. *"3 leakage columns identified and flagged for removal"*)
- [ ] Key finding 2
- [ ] Key finding 3

### 2. Cleaning — `02_clean.ipynb`
- Dropped 5 columns with no predictive value (`CustomerID`, `Count`,
  `Country`, `State`, `Lat Long`)
- Dropped 3 leakage columns (`Churn Score`, `Churn Label`, `Churn Reason`),
  with `Churn Reason` saved separately for later analysis
- [ ] Fixed `Total Charges` type from object → float (and how many rows were affected)

### 3. EDA — `03_eda.ipynb`
Charts and hypotheses about what drives churn.
- [ ] Top finding 1 (e.g. *"Month-to-month contracts churn at ~3x the rate of 2-year contracts"*)
- [ ] Top finding 2
- [ ] **Geographic finding** from the lat/long map — which regions churn most?
- [ ] Top finding 3

### 4. Preprocessing — `04_preprocess.ipynb`
- Decision on how to handle `City` (high-cardinality categorical)
- One-hot encoded categorical variables
- Scaled numerical features with `StandardScaler`
- Stratified 80/20 train/test split

### 5. Training & comparison — `05_train.ipynb`
Compared three models with 5-fold cross-validation:

| Model               | ROC-AUC | Precision | Recall | F1    |
|---------------------|---------|-----------|--------|-------|
| Logistic Regression | [ ]     | [ ]       | [ ]    | [ ]   |
| Random Forest       | [ ]     | [ ]       | [ ]    | [ ]   |
| **XGBoost** ⭐      | **[ ]** | **[ ]**   | **[ ]**| **[ ]**|

**Why ROC-AUC as primary metric:** the classes are imbalanced (~27% churn),
so plain accuracy is misleading. ROC-AUC measures how well the model ranks
churners above non-churners across all thresholds — which matches the actual
business use (rank customers by risk, target the top N for retention).

### 6. Evaluation — `06_evaluate.ipynb`
Final model on the held-out test set, with confusion matrix, classification
report, and SHAP-based explainability.

- [ ] Final test set ROC-AUC: **[ ]**
- [ ] Final test set F1: **[ ]**

### 7. Prediction example — `07_predict.ipynb`
Loads the saved model and runs a prediction on a single customer payload,
showing both probability and top SHAP drivers.

### 8. Bonus — `08_churn_reasons.ipynb`
A separate qualitative analysis of *why* customers churn, using the
`Churn Reason` column that was set aside. Shows top reasons broken down by
contract type, tenure, and customer segment — useful business context that
the model itself can't provide.

---

## 📈 Key findings

[ ] Fill in 3–5 most predictive features from SHAP analysis. For each, write
one sentence about what it means in business terms. Example:

> 1. **Contract type** — month-to-month contracts are by far the strongest
>    predictor of churn. Long-term contracts dramatically reduce risk.
> 2. **Tenure** — customers in their first year are at much higher risk;
>    risk drops sharply after 12–18 months.
> 3. **[ ]**

From the bonus churn-reasons analysis:
- [ ] Top reason customers actually gave for leaving
- [ ] How that reason varies across segments

---

## ⚠️ Limitations & honest tradeoffs

- The dataset is a **snapshot**, not time-series — the model can't capture
  how customer behavior changes over time before they churn.
- The dataset is **small** (~7k rows), so I avoided complex deep learning
  approaches that would over-fit.
- All customers are in **California** — the model's geographic findings
  shouldn't be generalized to other regions without retraining.
- [ ] Anything else you discover

---

## ▶️ How to run it locally

```bash
git clone https://github.com/[your-username]/customer-churn-predictor.git
cd customer-churn-predictor

python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

Then open the project in VS Code, select the venv as your Jupyter kernel,
and run the notebooks in order from `01_` through `07_` (with `08_` as an
optional bonus).

Dataset: [IBM Telco Churn (Kaggle)](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).
Drop the CSV into `data/raw/` and rename to `telco_churn.csv`.

---

## 💡 What I learned

[ ] Write 3–5 honest sentences after the project is done. Some prompts:

- The biggest unexpected challenge and how I worked through it
  (data leakage was a strong candidate here)
- A modeling decision I'd revisit with more time
- Something about ML I now understand that I didn't before
- What I'd add next

---

## 🚀 What's next (v2 ideas)

- Wrap the model in a **FastAPI** backend serving a `/predict` endpoint
- Build a **React** frontend with a form for customer details and a
  visualization of SHAP drivers
- Deploy backend to **Render** and frontend to **Vercel**
- Add a small "what-if" tool so business users can see how changing one
  variable (e.g. contract type) shifts a customer's churn probability
- An interactive map of California showing churn risk by region

---

## 👤 About me

Built by **[Akanksha Singh]**


