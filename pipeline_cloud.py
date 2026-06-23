import argparse
import warnings
import os
import boto3
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

def download_from_s3(bucket, key, local_path):
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)

def upload_to_s3(local_path, bucket, key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)

class Preprocessing:
    idCols = ['Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN']
    numericPlaceholderCols = ['Age', 'Annual_Income', 'Outstanding_Debt',
                                 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Monthly_Balance']
    ordinalCol = ['Credit_Mix']
    ordinalOrder = ['Bad', 'Standard', 'Good']

    def __init__(self):
        self.numCol = None
        self.catCol = None
        self.ordCol = self.ordinalCol
        self.nomCol = None
        self.transformer = None

    @staticmethod
    def _parse_history(val):
        if pd.isna(val): return np.nan
        try:
            parts = str(val).split(' ')
            return int(parts[0]) * 12 + int(parts[3])
        except Exception: return np.nan

    def clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(columns=[c for c in self.idCols if c in df.columns])
        for col in self.numericPlaceholderCols:
            df[col] = pd.to_numeric(df[col].astype(str).str.rstrip('_'), errors='coerce')
        df['Amount_invested_monthly'] = pd.to_numeric(df['Amount_invested_monthly'].astype(str).str.replace('_', '', regex=False), errors='coerce')
        df['Changed_Credit_Limit'] = pd.to_numeric(df['Changed_Credit_Limit'].replace('_', np.nan), errors='coerce')
        df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'].astype(str).str.replace('_', '', regex=False), errors='coerce')
        df['Occupation'] = df['Occupation'].replace('_______', np.nan)
        df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
        df['Payment_of_Min_Amount'] = df['Payment_of_Min_Amount'].replace('NM', np.nan)
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', np.nan)
        df.loc[(df['Age'] < 14) | (df['Age'] > 100), 'Age'] = np.nan
        df.loc[df['Interest_Rate'] > 40, 'Interest_Rate'] = np.nan
        df.loc[(df['Num_of_Loan'] == -100) | (df['Num_of_Loan'] > 10), 'Num_of_Loan'] = np.nan
        df.loc[(df['Num_Bank_Accounts'] < 0) | (df['Num_Bank_Accounts'] > 20), 'Num_Bank_Accounts'] = np.nan
        df.loc[df['Num_Credit_Card'] > 15, 'Num_Credit_Card'] = np.nan
        df.loc[df['Num_of_Delayed_Payment'] > 30, 'Num_of_Delayed_Payment'] = np.nan
        df.loc[df['Num_Credit_Inquiries'] > 20, 'Num_Credit_Inquiries'] = np.nan
        df.loc[df['Total_EMI_per_month'] > 5000, 'Total_EMI_per_month'] = np.nan
        df.loc[df['Monthly_Balance'] < -10000, 'Monthly_Balance'] = np.nan
        df['Credit_History_Months'] = df['Credit_History_Age'].apply(self._parse_history)
        df = df.drop(columns=['Credit_History_Age'])
        df['Loan_Type_Count'] = df['Type_of_Loan'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        df = df.drop(columns=['Type_of_Loan'])
        return df

    def fit(self, X: pd.DataFrame):
        self.numCol = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
        self.catCol = [c for c in X.columns if c not in self.numCol]
        self.nomCol = [c for c in self.catCol if c not in self.ordCol]
        numTransformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())])
        ordTransformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(categories=[self.ordinalOrder], handle_unknown='use_encoded_value', unknown_value=-1))])
        nomTransformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
        self.transformer = ColumnTransformer(transformers=[('num', numTransformer, self.numCol), ('ord', ordTransformer, self.ordCol), ('nom', nomTransformer, self.nomCol)], remainder='passthrough')
        self.transformer.fit(X)
        return self

    def transform(self, X: pd.DataFrame): return self.transformer.transform(X)
    def fit_transform(self, X: pd.DataFrame): return self.fit(X).transform(X)

class Training:
    modelRegistry = {
        'logistic_regression': lambda p: LogisticRegression(max_iter=1000, class_weight='balanced', **p),
        'decision_tree':       lambda p: DecisionTreeClassifier(class_weight='balanced', **p),
        'random_forest':       lambda p: RandomForestClassifier(class_weight='balanced', **p),
        'svm':                 lambda p: SVC(kernel='rbf', class_weight='balanced', **p),
        'gradient_boosting':   lambda p: GradientBoostingClassifier(**p),
        'xgboost':             lambda p: XGBClassifier(eval_metric='mlogloss', **p),
    }
    def __init__(self, modelName, params=None):
        self.modelName = modelName
        self.params = params or {}
        self.model = None
    def fit(self, xTrain, yTrain, sampleWeight=None):
        self.model = self.modelRegistry[self.modelName](self.params)
        if self.modelName in ('xgboost', 'gradient_boosting') and sampleWeight is not None:
            self.model.fit(xTrain, yTrain, sample_weight=sampleWeight)
        else:
            self.model.fit(xTrain, yTrain)
        return self.model

class Evaluation:
    def __init__(self, targetMapping, positiveClass='Poor'):
        self.targetMapping = targetMapping
        self.positiveClass = positiveClass
        self.positiveIndex = targetMapping[positiveClass]
    def evaluate(self, model, xTest, yTestEncoded) -> dict:
        preds = model.predict(xTest)
        report = classification_report(yTestEncoded, preds, output_dict=True, zero_division=0)
        return {'accuracy': report['accuracy'], 'macro_f1': report['macro avg']['f1-score'], 'poor_recall': report[str(self.positiveIndex)]['recall']}

modelConfigs = {
    'logistic_regression': {},
    'decision_tree':       {'random_state': 42},
    'random_forest':       {'n_estimators': 300, 'max_depth': 25, 'random_state': 42},
    'svm':                 {'random_state': 42},
    'gradient_boosting':   {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
    'xgboost':             {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42},
}

def run_pipeline(bucket, dataKey, modelS3Key, mappingS3Key, experimentName='creditScoreAssessment'):
    localCsv = 'data_D.csv'
    download_from_s3(bucket, dataKey, localCsv)
    
    raw = pd.read_csv(localCsv)
    preprocessing = Preprocessing()
    cleaned = preprocessing.clean_raw(raw)
    X = cleaned.drop(columns='Credit_Score')
    y = cleaned['Credit_Score']
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    targetMapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
    yTrainEnc, yTestEnc = yTrain.map(targetMapping), yTest.map(targetMapping)
    
    xTrainP = preprocessing.fit_transform(xTrain)
    xTestP = preprocessing.transform(xTest)
    evaluator = Evaluation(targetMapping, positiveClass='Poor')
    sampleWeights = compute_sample_weight('balanced', yTrainEnc)
    
    mlflow.set_experiment(experimentName)
    results, fittedModels = [], {}
    for modelName, params in modelConfigs.items():
        with mlflow.start_run(run_name=modelName):
            trainer = Training(modelName, params)
            model = trainer.fit(xTrainP, yTrainEnc, sampleWeight=sampleWeights)
            metrics = evaluator.evaluate(model, xTestP, yTestEnc)
            mlflow.log_metrics(metrics)
            fittedModels[modelName] = model
            results.append({'model': modelName, **metrics})
            
    bestName = pd.DataFrame(results).sort_values('macro_f1', ascending=False).iloc[0]['model']
    bestModel = fittedModels[bestName]
    fullPipeline = Pipeline(steps=[('preprocessing', preprocessing.transformer), ('classifier', bestModel)])
    fullPipeline.fit(X, y.map(targetMapping))
    
    joblib.dump(fullPipeline, 'final_model_pipeline.pkl')
    joblib.dump(targetMapping, 'target_mapping.pkl')
    upload_to_s3('final_model_pipeline.pkl', bucket, modelS3Key)
    upload_to_s3('target_mapping.pkl', bucket, mappingS3Key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--data-key', default='data_D.csv')
    parser.add_argument('--model-key', default='models/final_model_pipeline.pkl')
    parser.add_argument('--mapping-key', default='models/target_mapping.pkl')
    args = parser.parse_args()
    run_pipeline(args.bucket, args.data_key, args.model_key, args.mapping_key)
