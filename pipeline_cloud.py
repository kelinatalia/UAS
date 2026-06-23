"""
pipeline_cloud.py  —  soal 2.a

Sama persis dengan pipeline.py (local), dengan 2 tambahan:
  1. Download data_D.csv dari S3 sebelum training
  2. Upload final_model_pipeline.pkl + target_mapping.pkl ke S3 setelah training

Cara pakai (di EC2):
    python3 pipeline_cloud.py --bucket NAMA_BUCKET_LU --data-key data_D.csv
"""

import argparse
import warnings
import os
import tempfile
warnings.filterwarnings('ignore')

import boto3
import numpy as np
import pandas as pd
import joblib
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


# ------------------------------------------------------------------ #
# S3 helpers
# ------------------------------------------------------------------ #
def download_from_s3(bucket: str, key: str, local_path: str):
    print(f"[S3] Downloading s3://{bucket}/{key} -> {local_path}")
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_path)
    print(f"[S3] Download selesai.")


def upload_to_s3(local_path: str, bucket: str, key: str):
    print(f"[S3] Uploading {local_path} -> s3://{bucket}/{key}")
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)
    print(f"[S3] Upload selesai.")


# ------------------------------------------------------------------ #
# PREPROCESSING
# ------------------------------------------------------------------ #
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
        if pd.isna(val):
            return np.nan
        try:
            parts = str(val).split(' ')
            return int(parts[0]) * 12 + int(parts[3])
        except Exception:
            return np.nan

    def clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop(columns=[c for c in self.idCols if c in df.columns])

        for col in self.numericPlaceholderCols:
            df[col] = pd.to_numeric(df[col].astype(str).str.rstrip('_'), errors='coerce')

        df['Amount_invested_monthly'] = pd.to_numeric(
            df['Amount_invested_monthly'].astype(str).str.replace('_', '', regex=False), errors='coerce')
        df['Changed_Credit_Limit'] = pd.to_numeric(
            df['Changed_Credit_Limit'].replace('_', np.nan), errors='coerce')
        df['Monthly_Balance'] = pd.to_numeric(
            df['Monthly_Balance'].astype(str).str.replace('_', '', regex=False), errors='coerce')

        df['Occupation']           = df['Occupation'].replace('_______', np.nan)
        df['Credit_Mix']           = df['Credit_Mix'].replace('_', np.nan)
        df['Payment_of_Min_Amount']= df['Payment_of_Min_Amount'].replace('NM', np.nan)
        df['Payment_Behaviour']    = df['Payment_Behaviour'].replace('!@9#%8', np.nan)

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
        df['Loan_Type_Count'] = df['Type_of_Loan'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
        df = df.drop(columns=['Type_of_Loan'])
        return df

    def fit(self, X: pd.DataFrame):
        self.numCol = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
        self.catCol = [c for c in X.columns if c not in self.numCol]
        self.nomCol = [c for c in self.catCol if c not in self.ordCol]

        numTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        ordTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=[self.ordinalOrder],
                                        handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        nomTransformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.transformer = ColumnTransformer(transformers=[
            ('num', numTransformer, self.numCol),
            ('ord', ordTransformer, self.ordCol),
            ('nom', nomTransformer, self.nomCol)
        ], remainder='passthrough')
        self.transformer.fit(X)
        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ------------------------------------------------------------------ #
# TRAINING
# ------------------------------------------------------------------ #
class Training:
    modelRegistry = {
        'logistic_regression': lambda p: LogisticRegression(max_iter=1000, class_weight='balanced', **p),
        'decision_tree':       lambda p: DecisionTreeClassifier(class_weight='balanced', **p),
        'random_forest':       lambda p: RandomForestClassifier(class_weight='balanced', **p),
        'svm':                 lambda p: SVC(kernel='rbf', class_weight='balanced', **p),
        'gradient_boosting':   lambda p: GradientBoostingClassifier(**p),
        'xgboost':             lambda p: XGBClassifier(eval_metric='mlogloss', **p),
    }

    def __init__(self, modelName: str, params: dict = None):
        if modelName not in self.modelRegistry:
            raise ValueError(f"Unknown modelName '{modelName}'.")
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


# ------------------------------------------------------------------ #
# EVALUATION
# ------------------------------------------------------------------ #
class Evaluation:
    def __init__(self, targetMapping: dict, positiveClass: str = 'Poor'):
        self.targetMapping = targetMapping
        self.positiveClass = positiveClass
        self.positiveIndex = targetMapping[positiveClass]

    def evaluate(self, model, xTest, yTestEncoded) -> dict:
        preds = model.predict(xTest)
        report = classification_report(yTestEncoded, preds, output_dict=True, zero_division=0)
        return {
            'accuracy':   report['accuracy'],
            'macro_f1':   report['macro avg']['f1-score'],
            f'{self.positiveClass.lower()}_recall':    report[str(self.positiveIndex)]['recall'],
            f'{self.positiveClass.lower()}_precision': report[str(self.positiveIndex)]['precision'],
        }


# ------------------------------------------------------------------ #
# ORCHESTRATION
# ------------------------------------------------------------------ #
modelConfigs = {
    'logistic_regression': {},
    'decision_tree':       {'random_state': 42},
    'random_forest':       {'n_estimators': 300, 'max_depth': 25,
                             'min_samples_split': 2, 'min_samples_leaf': 1, 'random_state': 42},
    'svm':                 {'random_state': 42},
    'gradient_boosting':   {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
    'xgboost':             {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
                             'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42},
}


def run_pipeline(bucket: str, dataKey: str, experimentName: str = 'creditScoreAssessment',
                  modelS3Key: str = 'models/final_model_pipeline.pkl',
                  mappingS3Key: str = 'models/target_mapping.pkl'):

    # ---- 1. Download data dari S3 ----
    print(f"[1/6] Downloading data dari S3 ...")
    localCsv = 'data_D.csv'
    download_from_s3(bucket, dataKey, localCsv)

    # ---- 2. Clean ----
    print(f"[2/6] Cleaning data ...")
    raw = pd.read_csv(localCsv)
    preprocessing = Preprocessing()
    cleaned = preprocessing.clean_raw(raw)

    X = cleaned.drop(columns='Credit_Score')
    y = cleaned['Credit_Score']
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    targetMapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
    yTrainEnc = yTrain.map(targetMapping)
    yTestEnc  = yTest.map(targetMapping)

    # ---- 3. Preprocess ----
    print("[3/6] Fitting preprocessing (training fold only) ...")
    xTrainP = preprocessing.fit_transform(xTrain)
    xTestP  = preprocessing.transform(xTest)
    print(f"      -> train {xTrainP.shape}, test {xTestP.shape}")

    evaluator    = Evaluation(targetMapping, positiveClass='Poor')
    sampleWeights = compute_sample_weight('balanced', yTrainEnc)

    # ---- 4. Train & log ke MLflow ----
    print(f"[4/6] Training + logging ke MLflow (experiment='{experimentName}') ...")
    mlflow.set_experiment(experimentName)

    results, fittedModels = [], {}
    for modelName, params in modelConfigs.items():
        with mlflow.start_run(run_name=modelName):
            trainer = Training(modelName, params)
            model   = trainer.fit(xTrainP, yTrainEnc, sampleWeight=sampleWeights)
            metrics = evaluator.evaluate(model, xTestP, yTestEnc)

            mlflow.log_param('model_name', modelName)
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metrics({k: v for k, v in metrics.items()})
            mlflow.sklearn.log_model(model, artifact_path='model')

            fittedModels[modelName] = model
            results.append({'model': modelName, **metrics})
            print(f"      {modelName:20s} | acc={metrics['accuracy']:.4f} "
                  f"| macroF1={metrics['macro_f1']:.4f} | poorRecall={metrics['poor_recall']:.4f}")

    # ---- 5. Pilih best model ----
    print("[5/6] Memilih model terbaik ...")
    resultsDf = pd.DataFrame(results).sort_values('macro_f1', ascending=False).reset_index(drop=True)
    print(resultsDf.to_string(index=False))

    shortlist = resultsDf.head(2).reset_index(drop=True)
    bestRow   = shortlist.sort_values('poor_recall', ascending=False).iloc[0]
    bestName  = bestRow['model']
    bestModel = fittedModels[bestName]
    print(f"\n      Shortlist top-2 by macro F1: {list(shortlist['model'])}")
    print(f"      Best model (highest Poor recall): {bestName}")

    fullPipeline = Pipeline(steps=[
        ('preprocessing', preprocessing.transformer),
        ('classifier', bestModel)
    ])
    fullPipeline.fit(xTrain, yTrainEnc)

    # ---- 6. Upload model ke S3 ----
    print("[6/6] Upload model ke S3 ...")
    localModelPath   = 'final_model_pipeline.pkl'
    localMappingPath = 'target_mapping.pkl'
    joblib.dump(fullPipeline, localModelPath)
    joblib.dump(targetMapping, localMappingPath)

    upload_to_s3(localModelPath,   bucket, modelS3Key)
    upload_to_s3(localMappingPath, bucket, mappingS3Key)

    with mlflow.start_run(run_name=f'BEST_{bestName}'):
        mlflow.log_param('best_model', bestName)
        mlflow.log_metric('poor_recall', bestRow['poor_recall'])
        mlflow.log_metric('macro_f1',    bestRow['macro_f1'])
        mlflow.log_param('s3_model_path', f's3://{bucket}/{modelS3Key}')
        mlflow.sklearn.log_model(fullPipeline, artifact_path='best_full_pipeline')

    print(f"\nDone! Model tersimpan di s3://{bucket}/{modelS3Key}")
    return bestName


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket',     required=True, help='Nama S3 bucket lu')
    parser.add_argument('--data-key',   default='data_D.csv', help='S3 key untuk data_D.csv')
    parser.add_argument('--model-key',  default='models/final_model_pipeline.pkl')
    parser.add_argument('--mapping-key',default='models/target_mapping.pkl')
    parser.add_argument('--experiment', default='creditScoreAssessment')
    args = parser.parse_args()

    run_pipeline(args.bucket, args.data_key, args.experiment, args.model_key, args.mapping_key)
