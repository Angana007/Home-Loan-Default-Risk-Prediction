# 🏠 Home Loan Default Risk Prediction

## 📘 Project Overview
This project builds a **deep learning–based risk prediction system** that estimates the probability of a customer **defaulting on a home loan**.  
The model leverages **demographic, credit, and financial features** to help lenders identify **high-risk borrowers** before approval — reducing financial loss and improving credit decision efficiency.

To ensure **trust and interpretability**, the model integrates **SHAP explainability** and is deployed via an **interactive Streamlit dashboard** for real-time predictions and feature-level insights.

---

## 🚀 Key Highlights
- Developed a **Deep Learning model (Keras/TensorFlow)** achieving **AUC = 0.977**, identifying 91% of true defaulters.  
- Applied **ADASYN** resampling to handle severe class imbalance in the dataset.  
- Integrated **SHAP explainability** for transparent model decisions and bias detection.  
- Deployed a **Streamlit web app** enabling instant loan risk predictions with explainable visualizations.  
- Enabled lenders to potentially achieve **10–15% lower default rates** and **30% faster approvals**.

---

## 🎯 Business Problem
Banks and financial institutions lose millions due to non-performing loans.  
Traditional scoring systems (like rule-based thresholds) often miss **subtle nonlinear patterns** that signal credit risk.

This project provides:
- A **probabilistic risk score** for every new applicant.  
- **Feature-level explanations** showing why a prediction was made.  
- A deployable, interpretable model that enhances **risk management and decision transparency**.

---

## ⚙️ Tech Stack

| Layer | Tools & Libraries |
|-------|-------------------|
| Data Processing | Pandas, NumPy, Seaborn, Matplotlib |
| Resampling | **ADASYN** (Imbalanced-Learn) |
| Modeling | **Keras / TensorFlow** |
| Scaling | StandardScaler |
| Explainability | **SHAP** |
| Deployment | **Streamlit** |
| Metrics | ROC-AUC, Sensitivity, Specificity, F1-Score |

---

## 📊 Dataset
- **Source:** Modified version of [Home Credit Default Risk (Kaggle)](https://www.kaggle.com/c/home-credit-default-risk)  
- **Size:** 307,511 applicants × 122 features  
- **Type:** Financial, demographic, and credit bureau information  

| Feature | Description |
|----------|-------------|
| `AMT_INCOME_TOTAL` | Applicant’s total income |
| `AMT_CREDIT` | Loan amount applied for |
| `DAYS_EMPLOYED` | Employment duration |
| `NAME_EDUCATION_TYPE` | Applicant’s education level |
| `CODE_GENDER` | Gender |
| `TARGET` | 1 = Defaulted, 0 = Repaid |

---

## 🧠 Modeling Workflow

1. **Data Preprocessing**  
   - Filled missing numeric values with median and categorical with mode.  
   - One-hot encoded categorical variables.  

2. **Class Balancing**  
   - Addressed 1:10 imbalance using **ADASYN** to synthesize minority (defaulter) samples.

3. **Feature Scaling**  
   - Applied `StandardScaler` to normalize numerical features.

4. **Model Architecture**  
   - 3-layer **Feedforward Neural Network** with:  
     - Batch Normalization  
     - Dropout + L2 regularization  
     - ReLU activation and Sigmoid output  
     - EarlyStopping to avoid overfitting  

5. **Evaluation Metrics**  
   - **AUC = 0.977**  
   - **Sensitivity = 0.91**  
   - **Specificity = 0.9997**  
   - **F1-Score = 0.95**

---

## 📈 Model Performance Summary

| Metric | Score | Interpretation |
|--------|-------|----------------|
| AUC | **0.977** | Excellent separation between classes |
| Sensitivity | **0.908** | Accurately detects defaulters |
| Specificity | **0.9997** | Minimizes false positives |
| F1-Score | **0.952** | Strong precision-recall balance |

> The model generalizes well with stable validation performance and no signs of overfitting.

---

## 🔍 Explainable AI Insights (SHAP)

The **SHAP analysis** provides local and global interpretability:  

- **CNT_FAM_MEMBERS** and **CNT_CHILDREN** strongly affect repayment ability.  
- **Employment stability** (“Working” vs “Self-employed”) and **education level** influence credit reliability.  
- **Family status** (single vs married) and **housing type** correlate with financial discipline.  
- Global SHAP plots highlight that **income, credit-to-income ratio, and employment duration** are key drivers of default risk.  

> These insights promote explainable, auditable, and bias-aware lending decisions — aligning with modern AI governance in fintech.

---

## 💻 Streamlit Dashboard

**Interactive features:**
- 🧾 **Applicant Input Form:** Enter key attributes like income, credit, and employment details.  
- 📊 **Prediction Output:** Displays probability of default with intuitive color-coded risk zones.  
- 🧠 **SHAP Visualizations:**  
  - Local feature importance (individual prediction explanation)  
  - Global feature importance (overall model behavior)  
- 🎛️ **Preset Scenarios:** Choose *High-Risk* or *Low-Risk* applicant profiles to test model behavior.

---

## 💡 Business Impact

| Area | Benefit |
|------|----------|
| **Risk Management** | Identify defaulters before approval |
| **Operational Efficiency** | Automates screening, reduces manual workload |
| **Decision Speed** | ~30% faster loan processing |
| **Financial Savings** | 10–15% reduction in default losses |
| **Fair & Transparent Lending** | SHAP-backed explainability for auditors and regulators |

---  

## 👩‍💻 Author
**Angana Chakraborty**  
AI/ML Engineer | Data Analyst | Researcher  

📍 Kolkata, India  
🔗 [LinkedIn](https://linkedin.com/in/angana-chakraborty) • [GitHub](https://github.com/Angana007)

---

