# python
import os
import types
import pandas as pd
import numpy as np
import pytest
from types import SimpleNamespace

# Importer le module à tester (import absolu)
import model_pipeline as mp
from model_pipeline import (
    load_data, preprocess_data, create_features, remove_correlated_features,
    handle_outliers, prepare_training_data, initialize_models,
    train_and_evaluate_models
)
from sklearn.preprocessing import StandardScaler

def test_load_data_success(tmp_path):
    # créer deux petits csv
    df1 = pd.DataFrame({
        'A': [1,2],
        'Churn': [0,1]
    })
    df2 = pd.DataFrame({
        'A': [3,4],
        'Churn': [1,0]
    })
    p1 = tmp_path / "train.csv"
    p2 = tmp_path / "test.csv"
    df1.to_csv(p1, index=False)
    df2.to_csv(p2, index=False)
    X, y = load_data(str(p1), str(p2))
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert X.shape == (2, 2)
    assert y.shape == (2, 2)

def test_load_data_failure():
    X, y = load_data("nonexistent_file_123.csv", "also_missing_456.csv")
    assert X is None and y is None

def test_preprocess_data_basic():
    # construire DataFrames d'exemple
    X = pd.DataFrame({
        'Churn': [0,1],
        'International plan': ['No','Yes'],
        'Voice mail plan': ['Yes','No'],
        'State': ['NY','CA'],
        'Area code': [408,415],
        'Account length': [10,20]
    })
    y = X.copy()
    Xp, yp, encoders = preprocess_data(X, y)
    # 'State' et 'Area code' doivent avoir été supprimés
    assert 'State' not in Xp.columns
    assert 'Area code' not in Xp.columns
    # Les encoders doivent être retournés
    assert 'state_encoder' in encoders and 'area_encoder' in encoders
    # Binary maps
    assert set(Xp['International plan'].unique()) <= {0,1}
    assert set(Xp['Voice mail plan'].unique()) <= {0,1}

def test_create_features_values():
    X = pd.DataFrame({
        'Total day calls':[10],
        'Total eve calls':[5],
        'Total night calls':[2],
        'Total intl calls':[1],
        'Total day charge':[2.0],
        'Total eve charge':[1.0],
        'Total night charge':[0.5],
        'Total intl charge':[0.1],
        'Customer service calls':[2],
        'Account length':[10]
    })
    y = X.copy()
    Xf, yf = create_features(X.copy(), y.copy())
    assert 'Total calls' in Xf.columns
    assert 'Total charge' in Xf.columns
    assert 'CScalls Rate' in Xf.columns
    assert Xf.at[0,'Total calls'] == 10+5+2+1
    assert abs(Xf.at[0,'Total charge'] - (2.0+1.0+0.5+0.1)) < 1e-8
    assert Xf.at[0,'CScalls Rate'] == 2/10

def test_remove_correlated_features():
    cols = ['Total day minutes','Total eve minutes','Total night minutes','Total intl minutes','Voice mail plan','KeepMe']
    X = pd.DataFrame({c: [1,2] for c in cols})
    y = X.copy()
    Xr, yr = remove_correlated_features(X.copy(), y.copy())
    for c in ['Total day minutes','Total eve minutes','Total night minutes','Total intl minutes','Voice mail plan']:
        assert c not in Xr.columns
        assert c not in yr.columns
    assert 'KeepMe' in Xr.columns

def test_handle_outliers_iqr_branch(monkeypatch):
    # Forcer anderson -> non-normal (statistic > critical_values[2]) to hit IQR branch
    def fake_anderson(series):
        return SimpleNamespace(statistic=100.0, critical_values=[0,0,1,2,3])
    monkeypatch.setattr(mp, "anderson", fake_anderson)
    df = pd.DataFrame({
        'Account length': [1,2,3,1000],
        'Total day calls': [5,5,5,5]  # included in selected_columns
    })
    reduced = handle_outliers(df.copy())
    # extreme value 1000 should be removed
    assert reduced['Account length'].max() < 1000

def test_prepare_training_data_monkeypatched_smoteenn(monkeypatch):
    # Stub SMOTEENN to return inputs unchanged (but as numpy arrays)
    class StubSMOTEENN:
        def __init__(self, sampling_strategy=None, random_state=None):
            pass
        def fit_resample(self, X, y):
            # ensure returns numpy arrays
            return np.asarray(X), np.asarray(y)
    monkeypatch.setattr(mp, "SMOTEENN", StubSMOTEENN)
    # construire X,y avec Churn
    X = pd.DataFrame({
        'Churn': [0,1,0,1],
        'f1':[1,2,3,4],
        'f2':[0,1,0,1]
    })
    y = X.copy()
    X_train_scaled, X_test_scaled, y_resampled, y_test, scaler = prepare_training_data(X.copy(), y.copy(), sampling_strategy=0.5)
    # types
    assert isinstance(X_train_scaled, np.ndarray)
    assert isinstance(X_test_scaled, np.ndarray)
    assert isinstance(y_resampled, (np.ndarray, list))
    assert isinstance(y_test, pd.Series)
    assert isinstance(scaler, StandardScaler)

def test_initialize_models_keys():
    models = initialize_models()
    assert isinstance(models, dict)
    # check some expected model names
    for key in ["Logistic Regression","Random Forest","XGBoost","Neural Network"]:
        assert key in models

def test_train_and_evaluate_models_simple():
    # small synthetic train/test
    X_train = np.array([[0.],[1.],[0.],[1.]])
    y_train = np.array([0,1,0,1])
    X_test = np.array([[0.],[1.]])
    y_test = np.array([0,1])
    # use light models to speed up test
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    models = {
        "LR": LogisticRegression(),
        "KNN": KNeighborsClassifier(n_neighbors=1)
    }
    results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
    assert isinstance(results, dict)
    assert "LR" in results and "KNN" in results
    # check accuracy present or no error
    for k,v in results.items():
        assert isinstance(v, dict)
        assert 'accuracy' in v or 'error' in v

# End of tests