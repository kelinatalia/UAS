from pathlib import Path
import os
import boto3
import joblib
import pandas as pd
import streamlit as st

MODEL_LOCAL = Path(__file__).parent / 'final_model_pipeline.pkl'
MAPPING_LOCAL = Path(__file__).parent / 'target_mapping.pkl'

NUMERIC_FEATURES = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Months',
    'Loan_Type_Count'
]
CATEGORICAL_FEATURES = [
    'Month', 'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def _download_from_s3(bucket, s3_key, local_path):
    s3 = boto3.client(
        's3',
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=st.secrets["AWS_SESSION_TOKEN"],
        region_name=st.secrets["AWS_DEFAULT_REGION"]
    )
    s3.download_file(bucket, s3_key, str(local_path))

class InferenceService:
    def __init__(self, bucket=None):
        if not MODEL_LOCAL.exists() or not MAPPING_LOCAL.exists():
            if not bucket:
                bucket = st.secrets.get("S3_BUCKET")
            _download_from_s3(bucket, 'models/final_model_pipeline.pkl', MODEL_LOCAL)
            _download_from_s3(bucket, 'models/target_mapping.pkl', MAPPING_LOCAL)
            
        self.pipeline = joblib.load(MODEL_LOCAL)
        self.targetMapping = joblib.load(MAPPING_LOCAL)
        self.inverseMapping = {v: k for k, v in self.targetMapping.items()}

    def _validate(self, inputDict: dict):
        missing = [f for f in ALL_FEATURES if f not in inputDict]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

    def predict_one(self, inputDict: dict) -> dict:
        self._validate(inputDict)
        df = pd.DataFrame([{f: inputDict[f] for f in ALL_FEATURES}])
        predEncoded = self.pipeline.predict(df)[0]
        prediction = self.inverseMapping[predEncoded]
        result = {'prediction': prediction}
        if hasattr(self.pipeline, 'predict_proba'):
            proba = self.pipeline.predict_proba(df)[0]
            classes = self.pipeline.named_steps['classifier'].classes_
            result['probabilities'] = {
                self.inverseMapping[int(c)]: float(p) for c, p in zip(classes, proba)
            }
        return result

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [f for f in ALL_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        predsEncoded = self.pipeline.predict(df[ALL_FEATURES])
        df = df.copy()
        df['Prediction'] = [self.inverseMapping[p] for p in predsEncoded]
        return df

TEST_CASES = {
    'Good': {
        'Age': 37.0, 'Annual_Income': 45941.28, 'Monthly_Inhand_Salary': 3868.58,
        'Num_Bank_Accounts': 3, 'Num_Credit_Card': 4, 'Interest_Rate': 7,
        'Num_of_Loan': 2, 'Delay_from_due_date': 10, 'Num_of_Delayed_Payment': 8,
        'Changed_Credit_Limit': 6.76, 'Num_Credit_Inquiries': 3,
        'Outstanding_Debt': 716.97, 'Credit_Utilization_Ratio': 32.92,
        'Total_EMI_per_month': 60.85, 'Amount_invested_monthly': 164.06,
        'Monthly_Balance': 404.42, 'Credit_History_Months': 289, 'Loan_Type_Count': 2,
        'Month': 'July', 'Occupation': 'Lawyer', 'Credit_Mix': 'Good',
        'Payment_of_Min_Amount': 'No', 'Payment_Behaviour': 'High_spent_Medium_value_payments',
    },
    'Standard': {
        'Age': 33.0, 'Annual_Income': 36938.82, 'Monthly_Inhand_Salary': 3080.35,
        'Num_Bank_Accounts': 5, 'Num_Credit_Card': 5, 'Interest_Rate': 13,
        'Num_of_Loan': 3, 'Delay_from_due_date': 18, 'Num_of_Delayed_Payment': 14,
        'Changed_Credit_Limit': 10.25, 'Num_Credit_Inquiries': 5,
        'Outstanding_Debt': 998.81, 'Credit_Utilization_Ratio': 32.32,
        'Total_EMI_per_month': 62.77, 'Amount_invested_monthly': 137.15,
        'Monthly_Balance': 344.39, 'Credit_History_Months': 227, 'Loan_Type_Count': 3,
        'Month': 'January', 'Occupation': 'Lawyer', 'Credit_Mix': 'Standard',
        'Payment_of_Min_Amount': 'Yes', 'Payment_Behaviour': 'Low_spent_Small_value_payments',
    },
    'Poor': {
        'Age': 31.0, 'Annual_Income': 32123.85, 'Monthly_Inhand_Salary': 2628.98,
        'Num_Bank_Accounts': 7, 'Num_Credit_Card': 7, 'Interest_Rate': 21,
        'Num_of_Loan': 5, 'Delay_from_due_date': 27, 'Num_of_Delayed_Payment': 17,
        'Changed_Credit_Limit': 9.74, 'Num_Credit_Inquiries': 8,
        'Outstanding_Debt': 1954.62, 'Credit_Utilization_Ratio': 32.02,
        'Total_EMI_per_month': 74.87, 'Amount_invested_monthly': 116.73,
        'Monthly_Balance': 299.53, 'Credit_History_Months': 161, 'Loan_Type_Count': 5,
        'Month': 'June', 'Occupation': 'Mechanic', 'Credit_Mix': 'Bad',
        'Payment_of_Min_Amount': 'Yes', 'Payment_Behaviour': 'Low_spent_Small_value_payments',
    },
}
