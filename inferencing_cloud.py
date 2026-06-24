import os
import json
import boto3
import pandas as pd

ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'UAS-endpoint')
REGION        = os.environ.get('AWS_REGION', 'us-east-1')

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


class InferenceService:
    def __init__(self, endpoint_name: str = ENDPOINT_NAME, region: str = REGION):
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime', region_name=region)

    def _validate(self, input_dict):
        missing = [f for f in ALL_FEATURES if f not in input_dict]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

    def predict_one(self, input_dict: dict) -> dict:
        self._validate(input_dict)
        features = [input_dict[f] for f in ALL_FEATURES]
        payload = {"instances": [features]}

        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read().decode("utf-8"))

        label = result["labels"][0]
        probs = result["probabilities"][0]
        label_order = ['Poor', 'Standard', 'Good']

        return {
            'prediction': label,
            'probabilities': {l: p for l, p in zip(label_order, probs)}
        }
