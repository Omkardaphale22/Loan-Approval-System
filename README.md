# 🏦 Loan Approval Prediction System

A machine learning web app built with **Scikit-learn** and **Streamlit**.

---

## 📁 Project Structure

```
loan_approval_app/
│
├── loan_approval_data.csv    ← Your CSV data file (place it here)
├── train_model.py            ← Run once to train & save the model
├── app.py                    ← Streamlit UI (main app)
├── requirements.txt          ← Python dependencies
└── artifacts/                ← Auto-created after training
    ├── model.pkl
    ├── scaler.pkl
    ├── ohe.pkl
    ├── le_edu.pkl
    ├── le_target.pkl
    ├── num_imp.pkl
    ├── feature_names.pkl
    └── dropdown_options.pkl
```

---

## 🚀 Setup & Run (Step by Step)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Place your CSV file
Copy your `loan_approval_data.csv` into this folder (same folder as `app.py`).

### Step 3 — Train the model
```bash
python train_model.py
```
This will train a Logistic Regression model, print evaluation metrics,
and save all artifacts to the `artifacts/` folder.

### Step 4 — Launch the Streamlit app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your browser.

---

## 📋 CSV Column Reference

Your `loan_approval_data.csv` should have these columns:

| Column                | Type        | Example              |
|-----------------------|-------------|----------------------|
| Applicant_ID          | string      | APP001               |
| Gender                | categorical | Male / Female        |
| Marital_Status        | categorical | Married / Single     |
| Education_Level       | categorical | Graduate / Not Graduate |
| Employment_Status     | categorical | Employed / Self-Employed |
| Employer_Category     | categorical | Government / Private |
| Applicant_Income      | numeric     | 50000                |
| Co_Applicant_Income   | numeric     | 20000                |
| Loan_Amount           | numeric     | 200000               |
| Loan_Term             | numeric     | 120 (months)         |
| Credit_Score          | numeric     | 720                  |
| DTI_Ratio             | numeric     | 0.35                 |
| Loan_Purpose          | categorical | Home / Education     |
| Property_Area         | categorical | Urban / Rural        |
| Previous_Loan_Taken   | 0/1         | 1                    |
| Existing_Loans        | numeric     | 2                    |
| Loan_to_Value_Ratio   | numeric     | 0.8                  |
| Age                   | numeric     | 30                   |
| Number_of_Dependents  | numeric     | 2                    |
| Loan_Approved         | categorical | Yes / No             |

---

## 🧠 Model Info

- **Algorithm**: Logistic Regression
- **Features**: All above columns (with feature engineering: DTI²  and CreditScore²)
- **Preprocessing**: Mean imputation (numeric), Mode imputation (categorical), StandardScaler, OneHotEncoder

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set **Main file path** to `app.py`
5. Click **Deploy**

> ⚠️ For cloud deployment, you must commit the `artifacts/` folder too,
> OR add a startup script to retrain on first launch.
