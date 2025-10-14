# 🏠 Home Loan Default Risk Prediction

## 📘 Project Overview
This project develops a **deep learning model** to predict the likelihood of **home loan default** using customer demographics, credit history, and financial data.  
The model helps lenders identify high-risk borrowers before approval — reducing financial loss and improving decision-making efficiency.

To enhance **trust and interpretability**, the model is paired with **SHAP explainability** and deployed through an **interactive Streamlit dashboard** for real-time prediction and feature-level insights.

---

## 🎯 Business Objective
Financial institutions face substantial losses from loan defaults.  
Traditional rule-based credit scoring often fails to capture complex nonlinear relationships between income, credit behavior, and loan performance.  

This solution:
- Predicts the **probability of loan default** for new applicants.  
- Highlights **key factors influencing each decision** (e.g., credit score, loan-to-income ratio).  
- Enables **faster and more transparent approvals** via a business-friendly dashboard.

---

## ⚙️ Tech Stack
| Layer | Tools & Libraries |
|-------|-------------------|
| Data Processing | Pandas, NumPy, Seaborn, Matplotlib |
| Resampling | **ADASYN** (Imbalanced-Learn) |
| Modeling | **Keras / TensorFlow** |
| Scaling | StandardScaler |
| Explainability | **SHAP** |
| Deployment | **Streamlit Cloud** |
| Metrics | ROC-AUC, Sensitivity, Specificity, F1-Score |

---

## 📊 Dataset
- **Source:** Modified version of [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk)  
- **Rows:** 307,511 applicants  
- **Columns:** 122 financial, demographic, and credit features  

Example features:
| Feature | Description |
|----------|-------------|
| `AMT_INCOME_TOTAL` | Total applicant income |
| `AMT_CREDIT` | Loan amount requested |
| `CODE_GENDER` | Applicant gender |
| `NAME_EDUCATION_TYPE` | Education level |
| `DAYS_EMPLOYED` | Employment duration |
| `TARGET` | 1 = Defaulted, 0 = Repaid |

---

## 🧠 Modeling Workflow

1. **Data Cleaning & Imputation**
   - Filled missing numeric values with median  
   - Imputed categorical columns using mode  

2. **Encoding & Balancing**
   - One-hot encoded categorical variables  
   - Addressed severe class imbalance using **ADASYN** to synthesize minority (defaulter) samples  

3. **Feature Scaling**
   - Standardized all numeric features via `StandardScaler`  

4. **Deep Learning Model**
   - 3-layer **feed-forward neural network**  
   - Regularization: `Dropout` + `L2` penalty  
   - Batch Normalization for stable convergence  
   - Early stopping to prevent overfitting  

5. **Evaluation Metrics**
   - **AUC = 0.977** → Excellent discrimination ability  
   - **Sensitivity = 0.91** → Captures 91% of true defaulters  
   - **Specificity = 0.9997** → Minimizes false alarms  
   - **F1 = 0.95** → Strong precision-recall balance  

---

## 📈 Model Performance Summary

| Metric | Score | Interpretation |
|--------|-------|----------------|
| AUC | **0.977** | Excellent separation between classes |
| Sensitivity | **0.908** | Catches most true defaulters |
| Specificity | **0.9997** | Very few false positives |
| F1-Score | **0.952** | High precision & recall balance |

> The model demonstrates outstanding generalization, achieving ~95% accuracy without signs of overfitting.

---

## 💻 Streamlit Dashboard Features

The interactive app provides:
- 📋 **Applicant Input Form:** Enter key fields such as income, credit, employment length, and credit score.  
- 🧮 **Risk Score Output:** Predicts default probability with color-coded risk zone:
  - Low Risk (< 0.2)  
  - Medium Risk (0.2–0.5)  
  - High Risk (0.5–0.8)  
  - Critical (> 0.8)  
- 🔍 **SHAP Visualization:** Displays top factors increasing/decreasing risk for that applicant.  
- 📊 **Global Importance:** SHAP summary chart for all applicants.  
- 📂 **CSV Upload:** Batch risk scoring for loan portfolios.

---

## 💡 Business Impact

| Area | Benefit |
|------|----------|
| **Risk Management** | Identify high-risk borrowers early |
| **Operational Efficiency** | Automates applicant screening, saving analyst time |
| **Decision Speed** | Up to **30% faster** loan approvals |
| **Financial Savings** | Potential **10–15% reduction** in default rates |
| **Compliance** | Transparent, explainable model suitable for audits |

---
## 👩‍💻 Author
**Angana Chakraborty**  
AI/ML Engineer | Data Analyst | Researcher  

📍 Kolkata, India  
🔗 [LinkedIn](https://linkedin.com/in/angana-chakraborty) • [GitHub](https://github.com/Angana007)

---

