# CUSTOMER-CHURN-PREDICTOR

# Customer Churn Prediction

> A machine learning project that predicts whether a telecom customer is likely
> to churn, with explainable predictions powered by SHAP. Built end-to-end from
> raw data through model evaluation.

![Top churn drivers](outputs/figures/[shap_summary.png])

---

## 📌 Why this project

Customer churn costs telecom companies an estimated 15–25% of annual revenue,
and acquiring a new customer is roughly 5x more expensive than retaining an
existing one. A good churn model lets a business focus retention efforts on
the customers most likely to leave — and explainable predictions help the
business understand *why* a customer is at risk, not just that they are.

I built this project to learn the end-to-end machine learning workflow:
exploring real data, cleaning it, engineering features, comparing models,
evaluating honestly, and explaining predictions in a way a non-technical
stakeholder can act on.

---

## 📊 Dataset

**[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**
from Kaggle — 7,043 customers, 21 features.

- **Target:** \`Churn\` (Yes / No)
- **Class balance:** ~27% churners (imbalanced)
- **Features cover:** demographics (gender, senior citizen, partner), account
  info (tenure, contract type, monthly charges, payment method), and services
  used (phone, internet, streaming, etc.)

---

## 🛠️ Tech stack

**Language:** Python 3.12
**Data & ML:** pandas, numpy, scikit-learn, XGBoost, SHAP
**Visualization:** matplotlib, seaborn
**Environment:** Jupyter Notebooks in VS Code
**Version control:** Git + GitHub

---

## 📁 Project structure

\`\`\`
customer-churn-predictor/


├── data/

│   ├── raw/              # original Kaggle CSV

│   └── processed/        # cleaned data after Notebook 02


├── notebooks/

│   ├── 01_explore.ipynb        # first look at the data

│   ├── 02_clean.ipynb          # handling missing values, types

│   ├── 03_eda.ipynb            # exploration, charts, hypotheses

│   ├── 04_preprocess.ipynb     # encoding, scaling, train/test split

│   ├── 05_train.ipynb          # train + compare 3 models

│   ├── 06_evaluate.ipynb       # metrics, confusion matrix, SHAP

│   └── 07_predict.ipynb        # single-customer prediction example


├── outputs/

│   ├── figures/          # exported charts

│   └── models/           # trained model (.pkl)


├── .gitignore


├── requirements.txt


└── README.md
\`\`\`

---

## 🧠 Approach

### 1. Exploration — \`01_explore.ipynb\`
First look at shape, types, missing values, and target distribution.
- [ ] Key finding 1 (e.g. *"\`TotalCharges\` has 11 blank rows — all customers with 0 months of tenure"*)
- [ ] Key finding 2

### 2. Cleaning — \`02_clean.ipynb\`
- [ ] How I handled missing values
- [ ] Any type conversions needed (e.g. \`TotalCharges\` from object → float)

### 3. EDA — \`03_eda.ipynb\`
Charts and hypotheses about what drives churn.
- [ ] Top finding 1 (e.g. *"Month-to-month contracts churn at ~3x the rate of 2-year contracts"*)
- [ ] Top finding 2
- [ ] Top finding 3

### 4. Preprocessing — \`04_preprocess.ipynb\`
- One-hot encoded categorical variables
- Scaled numerical features with \`StandardScaler\`
- Stratified 80/20 train/test split

### 5. Training & comparison — \`05_train.ipynb\`
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

### 6. Evaluation — \`06_evaluate.ipynb\`
Final model on the held-out test set, with confusion matrix, classification
report, and SHAP-based explainability.

- [ ] Final test set ROC-AUC: **[ ]**
- [ ] Final test set F1: **[ ]**

### 7. Prediction example — \`07_predict.ipynb\`
Loads the saved model and runs a prediction on a single customer payload,
showing both probability and top SHAP drivers.

---

## 📈 Key findings

[ ] Fill in 3–5 most predictive features from SHAP analysis. For each, write
one sentence about what it means in business terms. Example:

> 1. **Contract type** — month-to-month contracts are by far the strongest
>    predictor of churn. Long-term contracts dramatically reduce risk.
> 2. **Tenure** — customers in their first year are at much higher risk;
>    risk drops sharply after 12–18 months.
> 3. **[ ]**

---

## ⚠️ Limitations & honest tradeoffs

- The dataset is a **snapshot**, not time-series — the model can't capture
  how customer behavior changes over time before they churn.
- The dataset is **small** (~7k rows), so I avoided complex deep learning
  approaches that would over-fit.
- [ ] Anything else you discover

---

## ▶️ How to run it locally

\`\`\`bash
git clone https://github.com/akanksha726/customer-churn-predictor.git
cd customer-churn-predictor

python -m venv venv
venv\\Scripts\\activate            # Windows
# source venv/bin/activate         # Mac/Linux

pip install -r requirements.txt
\`\`\`

Then open the project in VS Code (or Jupyter), select the venv as your
kernel, and run the notebooks in order from \`01_\` through \`07_\`.

The Telco dataset is included in \`data/raw/\`. If you want to re-download it,
get it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## 💡 What I learned

[ ] Write 3–5 honest sentences after the project is done. Some prompts:

- The biggest unexpected challenge and how I worked through it
- A modeling decision I'd revisit with more time
- Something about ML I now understand that I didn't before
- What I'd add next (deployment as a web app, retraining pipeline,
  monitoring for data drift, etc.)

---

## 🚀 What's next (v2 ideas)

- Wrap the model in a **FastAPI** backend serving a \`/predict\` endpoint
- Build a **React** frontend with a form for customer details and a
  visualization of SHAP drivers
- Deploy backend to **Render** and frontend to **Vercel**
- Add a small "what-if" tool so business users can see how changing one
  variable (e.g. contract type) shifts a customer's churn probability

---

