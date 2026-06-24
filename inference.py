import os
import json
import joblib
import numpy as np
import pandas as pd

JSON_CONTENT_TYPE = "application/json"

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

LABEL_MAP = {0: 'Poor', 1: 'Standard', 2: 'Good'}


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "final_model_pipeline.pkl"))


def input_fn(request_body, request_content_type):
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        if isinstance(instances[0], dict):
            return pd.DataFrame(instances, columns=ALL_FEATURES)
        return pd.DataFrame(instances, columns=ALL_FEATURES)
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    probs = model.predict_proba(input_data)
    class_ids = np.argmax(probs, axis=1)
    labels = [LABEL_MAP[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }


def output_fn(prediction, accept_content_type):
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE
    raise ValueError(f"Unsupported accept type: {accept_content_type}")
