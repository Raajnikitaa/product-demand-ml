# DemandWise — Smart Retail Demand Predictor

DemandWise is a complete, beginner-friendly retail demand surge prediction web application built with **Python**, **Streamlit**, **Pandas**, **Scikit-learn**, **Plotly**, and **SQLite**.

The system helps retail shopkeepers anticipate which product categories are likely to experience an exceptional demand spike over the upcoming 7 days, allowing them to prepare buffer stock and avoid costly stockouts.

> **Public repository contents:** This repository publishes source code, tests, dependency instructions, and a pre-trained model pipeline with its metadata. The model was trained on the synthetic demonstration workbooks described below; those workbooks, user uploads, and SQLite databases are not included. The app can use the published model to score data supplied by a user, but bundled demo stores are unavailable unless their workbooks are provided separately. GitHub hosts this code; Streamlit Community Cloud runs the live application.

---

## 📌 Project Overview & Problem Statement

In multi-store retail, sudden product sales surges cause stockouts, lost revenue, and emergency freight expenses. Rather than forecasting noisy raw sales volumes, DemandWise solves the operational decision problem:

> **Will this product category in this store experience an abnormal demand surge (≥ 30% increase) over the next 7 days?**

This binary classification signal serves as an operational alert for shopkeepers to:
1. Review safety stock and supplier replenishment lead times.
2. Verify upcoming promotional schedules and supplier delivery buffers.
3. Prepare staff and warehouse allocations before the surge occurs.

---

## 📊 Dataset & Integrity Disclosure

### 1. Demonstration Workbooks
The application is pre-configured with five Excel workbooks located in `data/`:
- `DemandWise_Store_01_Central.xlsx` (Downtown Flagship)
- `DemandWise_Store_02_Market.xlsx` (City Market Supermarket)
- `DemandWise_Store_03_Residential.xlsx` (Green Valley Neighborhood Store)
- `DemandWise_Store_04_Suburban.xlsx` (West Pines Suburban Retail)
- `DemandWise_Store_05_HighStreet.xlsx` (High Street Shopping Mall)

### 2. Nature of Data: Synthetic Demonstration Records
> [!IMPORTANT]
> These workbooks contain **synthetic / simulated daily retail records** created for software testing, multi-store upload demonstration, and machine learning pipeline evaluation. They cover **January 1, 2025 through October 16, 2026** (654 days × 8 categories = 5,232 rows per store; **26,160 rows total**). They are **not real business transactions** and must not be cited as real-world market evidence.

### 3. What a Row Represents
Each row in the `Sales_Data` worksheet represents the daily sales activity of **one product category** in **one store** on **one date**:
- `date`: Daily sales timestamp
- `store_name`: Retail branch identifier
- `product`: Category-level assortment label (e.g. *Fresh Produce Assortment*)
- `category`: Monitored category (e.g. *Groceries, Beverages, Personal Care, Household Cleaning, Snacks, Dairy, Fresh Produce, Bakery*)
- `sales`: Units sold on that day
- `unit_price`: Illustrative unit price ($)
- `revenue`: Gross sales ($ = \text{sales} \times \text{unit\_price}$)
- `promotion`: Binary campaign indicator (1 if promotion active, 0 otherwise)

---

## 📐 Mathematical Formulation of Demand Surge

For a given store and product category evaluated on prediction date $t$:

$$\text{Surge}_t = 1 \iff \overline{\text{Sales}}_{[t+1, t+7]} \ge 1.30 \times \overline{\text{Sales}}_{[t-6, t]} \quad \text{AND} \quad \overline{\text{Sales}}_{[t+1, t+7]} > \overline{\text{Sales}}_{[t-6, t]} \quad \text{AND} \quad \overline{\text{Sales}}_{[t-6, t]} > 0$$

- **Historical Baseline ($\overline{\text{Sales}}_{[t-6, t]}$):** Average daily sales over the trailing 7 days up to and including day $t$.
- **Forward Horizon ($\overline{\text{Sales}}_{[t+1, t+7]}$):** Average daily sales over the upcoming 7 days starting from $t+1$.
- **30% Threshold:** A configurable business rule used to label a surge; it is not a claim about typical retail fluctuations.
- **Positive Baseline:** A target is assigned only when the preceding 7-day baseline is positive.
- **Incomplete Windows:** Rows without all seven baseline days or all seven future days are assigned `target = NaN` and excluded from training.

---

## 🛡️ Preventing Data Leakage & 7-Day Purge Gaps

1. **Information Horizon:** All rolling features (`past_7d_avg`, `past_14d_avg`, `past_30d_avg`, `past_7d_std`, `sales_7d_ago`, `sales_vs_7d_avg`, `promo_7d`) are calculated **strictly on data $\le t$**.
2. **Target Horizon:** The target is calculated strictly on forward sales $[t+1 \dots t+7]$ and dropped from model input features.
3. **Chronological Splitting with 7-Day Purge Gaps:**
   ```text
   [ Train Core: 2025-01-31 to 2026-06-27 ]  (20,520 rows)
                     │
         [ 7-day Purge Gap: 2026-06-28 to 2026-07-04 ]
                     │
   [ Validation Slice: 2026-07-05 to 2026-08-03 ]  (1,200 rows)
                     │
         [ 7-day Purge Gap: 2026-08-04 to 2026-08-10 ]
                     │
   [ Holdout Test Set: 2026-08-11 to 2026-10-09 ]  (2,400 rows)
   ```
   *Why are purge gaps mandatory?* Because training day $T_{train}$ uses forward sales up to $T_{train}+7$. Without a 7-day purge gap, training targets would overlap and leak into validation or test evaluation periods.
4. **Validation Model and Threshold Selection:** Majority-class, balanced Logistic Regression, and unweighted Logistic Regression candidates use the same training and validation periods. The model and threshold are selected using validation F1 only (ties prefer higher precision, then a higher threshold). Only the selected classifier is evaluated on the untouched test period; the majority baseline is reported as a reference.

---

## 📈 Model Architecture & Actual Measured Evaluation

### Pipeline Architecture
- **Numeric Pipeline:** `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`
- **Categorical Pipeline:** `SimpleImputer(strategy='constant', fill_value='Unknown')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`
- **Classifier:** Logistic Regression; the class-weight setting is selected by validation comparison (`None` / unweighted or `balanced`).

### Actual Measured Holdout Test Metrics (2,400 records)

The saved model was retrained on the five listed synthetic demonstration workbooks. The uploaded surge-test workbook was not included. These metrics apply to the chronological holdout from the demo datasets and do not establish accuracy on an independent retailer's data.

| Evaluation Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Selected model** | Unweighted Logistic Regression | Chosen by validation F1 |
| **Decision threshold ($\tau^*$)** | **0.164** | Selected on validation only |
| **Accuracy** | **97.08%** | Majority baseline is 97.83% |
| **Precision** | **15.38%** | 4 of 26 positive predictions were surges |
| **Recall** | **7.69%** | 4 of 52 actual surges detected |
| **F1-Score** | **0.1026** | Still low; surge detection remains weak |
| **ROC-AUC** | **0.5693** | Weak separation |
| **PR-AUC** | **0.0769** | Test surge prevalence is 0.0217 |
| **Brier Score** | **0.0217** | Probability-quality diagnostic; probabilities are not calibrated |
| **Confusion Matrix** | **TN = 2,326 \| FP = 22 \| FN = 48 \| TP = 4** | Total = 2,400 examples |
| **Surge Class Rate** | **2.17%** | 52 surges vs 2,348 non-surges |

Validation had 35 positive examples out of 1,200, so threshold selection is flagged as unstable. Balanced Logistic Regression had validation F1 0.0851; unweighted Logistic Regression achieved 0.1132. On the test set, the majority baseline gets 97.83% accuracy but detects no surges (F1 = 0). The selected model improves F1 versus the previous saved model (0.0522 to 0.1026), but detects only 4 of 52 surges and remains unsuitable for high-stakes decisions without more representative, independently validated data.

The uploaded `DemandWise_Surge_Test_Dataset.xlsx` validated at 5,232 rows, with dates 2025-01-01 through 2026-10-16 and eight categories. Its own chronological split had no surge labels in training or validation and 51 in the test partition, so it is not suitable as the sole training source. Uploaded workbooks are used for inference; they enter model training only when explicitly selected in the Model Dashboard.

### Download the Trained Model Bundle
Open **Model & Report** in the dashboard and select **Download Model, Graphs & Code**. The ZIP includes the serialized trained pipeline, evaluation metadata, a standalone HTML report with graphs and model settings, and the training source files.

---

## 🚀 Quick Start Guide (Windows)

### 1. Prerequisites
- Python 3.11+ (verified on Python 3.14.3)
- PowerShell or Windows Command Prompt

### 2. Activate Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 4. Run Automated Test Suite
```powershell
python -m pytest -q
```
*Latest verified result: 42 passed tests.*

### 5. Train / Retrain Model
```powershell
python -m src.train_model
```

### 6. Launch the Web Application
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

Launch the separate model and evaluation dashboard with:
```powershell
python -m streamlit run model_dashboard.py
```

---

## 🏗️ Project Directory Structure

```text
Product-Demand-Surge-Prediction1/
├── app.py                     # Main 6-section Streamlit application
├── requirements.txt           # Pinned project dependencies
├── pytest.ini                 # Pytest configuration with pythonpath
├── README.md                  # Comprehensive documentation and viva guide
├── .gitignore                 # Cache and environment ignore rules
├── data/                      # 5 synthetic Excel workbooks (5,232 rows each)
│   ├── DemandWise_Store_01_Central.xlsx
│   ├── DemandWise_Store_02_Market.xlsx
│   ├── DemandWise_Store_03_Residential.xlsx
│   ├── DemandWise_Store_04_Suburban.xlsx
│   ├── DemandWise_Store_05_HighStreet.xlsx
│   └── README_Dataset_Guide.txt
├── database/
│   └── demandwise.db          # SQLite database (users, business profiles, stores)
├── models/
│   ├── model_pipeline.joblib  # Serialized Scikit-learn Pipeline
│   └── model_metadata.json    # Genuine training metadata and evaluation report
├── src/
│   ├── __init__.py            # Package marker
│   ├── auth.py                # PBKDF2 password hashing and user auth
│   ├── database.py            # SQLite queries with parameterized statements
│   ├── data_loader.py         # Excel loading and complete calendar reindexing
│   ├── data_validation.py     # Schema checking, date & duplicate validation
│   ├── feature_engineering.py # Leakage-free features & exact surge target
│   ├── preprocessing.py       # ColumnTransformer & Scikit-learn Pipeline
│   ├── evaluate_model.py      # Classification metrics & evaluation curves
│   ├── train_model.py         # Chronological training & artifact saving
│   ├── predict_demand.py      # Inference & category surge probability
│   ├── recommendations.py     # Plain-language explanations & retail advisories
│   └── utils.py               # SaaS Plotly charts (white/neutral theme)
└── tests/
    ├── __init__.py            # Tests package marker
    ├── test_data_validation.py      # Schema & invalid Excel tests
    ├── test_feature_engineering.py  # Leakage prevention & formula tests
    └── test_model_workflow.py       # DB, auth & model prediction tests
```

---

## 🎓 Academic Viva & Project Presentation Guide

### Q1: Why is this framed as a classification task instead of a regression sales forecast?
**Answer:** Daily retail sales exhibit severe day-of-week seasonality, irregular transactions, and random noise. Forecasting exact continuous volume ($R^2$ formulation) often fails operationally. Instead, inventory managers need an **operational triage signal**: *Will this category surge by ≥ 30% next week?* This binary signal directly triggers buffer stock escalation, supplier lead-time checks, and staffing preparations.

### Q2: Why is $R^2$ inappropriate for this model?
**Answer:** $R^2$ (coefficient of determination) evaluates continuous variance in regression models. It is mathematically undefined and meaningless for binary classification targets ($\{0, 1\}$). Precision, Recall, F1-score, and ROC-AUC are the correct academic metrics.

### Q3: How is data leakage rigorously prevented?
**Answer:**
1. **Information Horizon:** All retrospective features (rolling means, std dev, ratios, lags) use data strictly $\le t$.
2. **Target Horizon:** The target uses forward observations $[t+1 \dots t+7]$ and is never included in features.
3. **Purge Gaps:** Chronological splits enforce explicit 7-day buffers between training, validation, and test periods.
4. **Validation Tuning:** Decision threshold $\tau^*$ is chosen on the validation slice, not the test set.

### Q4: Why can't the app directly calculate exact inventory order quantities?
**Answer:** Generating purchase orders requires warehouse on-hand inventory balances, minimum order quantities (MOQ), supplier lead times, and shelf capacities. The dataset tracks checkout volume, not warehouse inventory. DemandWise honestly reports demand surge likelihood rather than fabricating ungrounded reorder advice.
