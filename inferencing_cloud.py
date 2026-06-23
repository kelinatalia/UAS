import json
import os
import joblib
import numpy as np
import pandas as pd

jsonContentType = "application/json"
csvContentType = "text/csv"

numericFeatures = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
    'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Months',
    'Loan_Type_Count'
]

categoricalFeatures = [
    'Month', 'Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'
]

allFeatures = numericFeatures + categoricalFeatures


def model_fn(modelDir: str):
    pipelinePath = os.path.join(modelDir, "final_model_pipeline.pkl")
    mappingPath = os.path.join(modelDir, "target_mapping.pkl")

    pipeline = joblib.load(pipelinePath)
    targetMapping = joblib.load(mappingPath)
    inverseMapping = {v: k for k, v in targetMapping.items()}

    return {
        "pipeline": pipeline,
        "inverseMapping": inverseMapping
    }


def input_fn(requestBody, requestContentType: str) -> pd.DataFrame:
    if requestContentType == jsonContentType:
        payload = json.loads(requestBody)
        instances = payload["instances"]
        return pd.DataFrame(instances, columns=allFeatures)

    if requestContentType == csvContentType:
        if isinstance(requestBody, (bytes, bytearray)):
            requestBody = requestBody.decode("utf-8")
        rows = [
            [x.strip() for x in line.split(",")]
            for line in requestBody.strip().splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows, columns=allFeatures)

    raise ValueError(f"unsupported content type: {requestContentType}")


def predict_fn(inputData: pd.DataFrame, modelArtifacts: dict) -> dict:
    pipeline = modelArtifacts["pipeline"]
    inverseMapping = modelArtifacts["inverseMapping"]

    predsEncoded = pipeline.predict(inputData)
    predictions = [inverseMapping[p] for p in predsEncoded]

    result = {
        "predictions": predictions
    }

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(inputData)
        classes = pipeline.named_steps["classifier"].classes_
        
        probsList = []
        for rowProbs in probs:
            rowDict = {inverseMapping[int(c)]: float(p) for c, p in zip(classes, rowProbs)}
            probsList.append(rowDict)
        result["probabilities"] = probsList

    return result


def output_fn(prediction: dict, acceptContentType: str):
    if acceptContentType == jsonContentType:
        return json.dumps(prediction), jsonContentType
    raise ValueError(f"unsupported accept type: {acceptContentType}")
