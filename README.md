## ▶️ Run the App

```bash
streamlit run src/app.py

**Patient Readmission Risk Prediction (Healthcare ML Project)**
**Overview**
This project predicts whether a patient will be readmitted within 30 days using a real-world hospital dataset. It demonstrates an end-to-end machine learning workflow, from data preprocessing to deployment via an interactive web application.

**Objective**
To identify patients at risk of 30-day readmission and explore how machine learning can support early risk detection in healthcare settings.

**Dataset**
- Diabetes dataset from 130 US hospitals
- ~70,000 patient records
- Imbalanced target (~9% readmitted within 30 days)

**Project Workflow**
1. Data Preprocessing
  - Handled missing values (? → null)
  - Removed duplicate patient records
  - Feature selection and cleaning
2. Exploratory Data Analysis
  - Target distribution analysis
  - Demographic and clinical feature visualization
3. Modeling
  - Logistic Regression (baseline and class-balanced)
  - Random Forest (comparison model)
4. Key Challenge: Class Imbalance
  - Dataset heavily skewed (~9% positive class)
  - Accuracy alone was misleading
  - Focus shifted to recall (sensitivity)

**Results**
Logistic Regression (Balanced)
  - Recall (readmitted): 0.54–0.57
  - Accuracy: ~0.55–0.63
  - Better at detecting high-risk patients
Random Forest
  - Accuracy: ~0.91
  - Failed to detect minority class without tuning

Key Insights
  - Higher number of lab procedures and medications → increased risk
  - Diagnosis complexity strongly influences readmission
  - Longer hospital stays associated with higher risk
  - Older age groups show increased likelihood of readmission

**Interactive App**
A Streamlit app was developed to simulate readmission risk prediction using selected features.

Features:
  - User input for patient characteristics
  - Real-time risk prediction
  - Simple risk explanation based on input factors
  - Probability-based risk categorization (Low / Moderate / High)

**Disclaimer**
This is a simplified portfolio project and is not intended for clinical use. The model is trained on limited features and serves only as a demonstration of machine learning and deployment skills.

Tech Stack
  - Python
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Streamlit

**Key Takeaway**
This project highlights the importance of handling imbalanced data and prioritizing recall in healthcare-related machine learning problems, where missing high-risk patients can have serious consequences.
