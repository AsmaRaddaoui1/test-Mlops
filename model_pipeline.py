"""
Pipeline modulaire pour la prédiction de churn client
Auteur: []
Date: []
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import anderson, zscore

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, log_loss, cohen_kappa_score, 
    matthews_corrcoef
)

# Optimization
import optuna

def load_data(train_path, test_path):
    """
    Charge les données d'entraînement et de test
    
    Parameters:
    train_path (str): Chemin vers le fichier d'entraînement
    test_path (str): Chemin vers le fichier de test
    
    Returns:
    tuple: (X_train, y_train) - DataFrames contenant les données
    """
    try:
        X = pd.read_csv(train_path)
        y = pd.read_csv(test_path)
        print(f"✅ Données chargées avec succès")
        print(f"   - Train shape: {X.shape}")
        print(f"   - Test shape: {y.shape}")
        return X, y
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        return None, None

def explore_data(X, y):
    """
    Explore les données avec des statistiques descriptives
    
    Parameters:
    X (DataFrame): Données d'entraînement
    y (DataFrame): Données de test
    
    Returns:
    tuple: (X, y) - Les DataFrames inchangés
    """
    print("\n🔍 Exploration des données:")
    print("=" * 50)
    
    # Informations de base
    print(f"Shape X: {X.shape}")
    print(f"Shape y: {y.shape}")
    
    # Types de données
    print("\nTypes de données (X):")
    print(X.dtypes.value_counts())
    
    # Valeurs manquantes
    print("\nValeurs manquantes (X):")
    print(X.isna().sum().sum())
    
    # Distribution de la target
    if 'Churn' in X.columns:
        print(f"\nDistribution de Churn (Train):")
        print(X['Churn'].value_counts())
    if 'Churn' in y.columns:
        print(f"\nDistribution de Churn (Test):")
        print(y['Churn'].value_counts())
    
    return X, y

def preprocess_data(X, y):
    """
    Prétraite les données (encodage, nettoyage)
    
    Parameters:
    X (DataFrame): Données d'entraînement
    y (DataFrame): Données de test
    
    Returns:
    tuple: (X_processed, y_processed, encoders) - Données prétraitées et encodeurs
    """
    print("\n🔄 Prétraitement des données...")
    
    # Créer des copies pour éviter les modifications originales
    X_processed = X.copy()
    y_processed = y.copy()
    
    # Conversion des types
    X_processed['Churn'] = X_processed['Churn'].astype(int)
    y_processed['Churn'] = y_processed['Churn'].astype(int)
    
    # Encodage des variables binaires
    X_processed['International plan'] = X_processed['International plan'].map({'No': 0, 'Yes': 1})
    y_processed['International plan'] = y_processed['International plan'].map({'No': 0, 'Yes': 1})
    
    X_processed['Voice mail plan'] = X_processed['Voice mail plan'].map({'No': 0, 'Yes': 1})
    y_processed['Voice mail plan'] = y_processed['Voice mail plan'].map({'No': 0, 'Yes': 1})
    
    # Encodage One-Hot pour State et Area code
    encoder_state = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_area = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit et transform sur les données d'entraînement
    encoded_states_X = encoder_state.fit_transform(X_processed[['State']])
    encoded_area_X = encoder_area.fit_transform(X_processed[['Area code']])
    
    # Transform sur les données de test
    encoded_states_y = encoder_state.transform(y_processed[['State']])
    encoded_area_y = encoder_area.transform(y_processed[['Area code']])
    
    # Création des DataFrames encodés
    encoded_states_df_X = pd.DataFrame(
        encoded_states_X, 
        columns=encoder_state.get_feature_names_out(['State'])
    )
    encoded_states_df_y = pd.DataFrame(
        encoded_states_y, 
        columns=encoder_state.get_feature_names_out(['State'])
    )
    
    encoded_area_df_X = pd.DataFrame(
        encoded_area_X, 
        columns=encoder_area.get_feature_names_out(['Area code'])
    )
    encoded_area_df_y = pd.DataFrame(
        encoded_area_y, 
        columns=encoder_area.get_feature_names_out(['Area code'])
    )
    
    # Suppression des colonnes originales et concaténation
    X_processed = X_processed.drop(['State', 'Area code'], axis=1)
    y_processed = y_processed.drop(['State', 'Area code'], axis=1)
    
    X_processed = pd.concat([X_processed, encoded_states_df_X, encoded_area_df_X], axis=1)
    y_processed = pd.concat([y_processed, encoded_states_df_y, encoded_area_df_y], axis=1)
    
    encoders = {
        'state_encoder': encoder_state,
        'area_encoder': encoder_area
    }
    
    print(f"✅ Prétraitement terminé")
    print(f"   - X shape après prétraitement: {X_processed.shape}")
    print(f"   - y shape après prétraitement: {y_processed.shape}")
    
    return X_processed, y_processed, encoders

def handle_outliers(X):
    """
    Gère les outliers selon le type de distribution
    
    Parameters:
    X (DataFrame): Données à traiter
    
    Returns:
    DataFrame: Données sans outliers
    """
    print("\n📊 Gestion des outliers...")
    
    selected_columns = [
        'Account length', 'Total day minutes', 'Total day calls', 'Total day charge',
        'Total eve minutes', 'Total eve calls', 'Total eve charge',
        'Total night minutes', 'Total night calls', 'Total night charge',
        'Total intl minutes', 'Total intl calls', 'Total intl charge'
    ]
    
    # Vérifier quelles colonnes existent dans X
    available_columns = [col for col in selected_columns if col in X.columns]
    
    # Séparation basée sur le test Anderson-Darling
    selected_normal_columns = []
    selected_other_columns = []
    
    for column in available_columns:
        try:
            result = anderson(X[column])
            if result.statistic < result.critical_values[2]:
                selected_normal_columns.append(column)
            else:
                selected_other_columns.append(column)
        except:
            selected_other_columns.append(column)
    
    print(f"   - Colonnes normales: {len(selected_normal_columns)}")
    print(f"   - Colonnes non-normales: {len(selected_other_columns)}")
    
    # Méthode Z-Score pour les distributions normales
    if selected_normal_columns:
        z_scores = zscore(X[selected_normal_columns])
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        X = X[filtered_entries]
        print(f"   - Outliers Z-score supprimés: {len(X)} échantillons restants")
    
    # Méthode IQR pour les autres distributions
    def remove_outliers_iqr(df, columns):
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df
    
    if selected_other_columns:
        X = remove_outliers_iqr(X, selected_other_columns)
        print(f"   - Outliers IQR supprimés: {len(X)} échantillons restants")
    
    return X

def create_features(X, y):
    """
    Crée de nouvelles features
    
    Parameters:
    X (DataFrame): Données d'entraînement
    y (DataFrame): Données de test
    
    Returns:
    tuple: (X, y) - DataFrames avec nouvelles features
    """
    print("\n🎯 Création de nouvelles features...")
    
    # Features pour X
    X['Total calls'] = (X['Total day calls'] + X['Total eve calls'] + 
                       X['Total night calls'] + X['Total intl calls'])
    X['Total charge'] = (X['Total day charge'] + X['Total eve charge'] + 
                        X['Total night charge'] + X['Total intl charge'])
    X['CScalls Rate'] = X['Customer service calls'] / X['Account length']
    
    # Features pour y
    y['Total calls'] = (y['Total day calls'] + y['Total eve calls'] + 
                       y['Total night calls'] + y['Total intl calls'])
    y['Total charge'] = (y['Total day charge'] + y['Total eve charge'] + 
                        y['Total night charge'] + y['Total intl charge'])
    y['CScalls Rate'] = y['Customer service calls'] / y['Account length']
    
    print("✅ Nouvelles features créées: Total calls, Total charge, CScalls Rate")
    
    return X, y

def remove_correlated_features(X, y):
    """
    Supprime les features corrélées
    
    Parameters:
    X (DataFrame): Données d'entraînement
    y (DataFrame): Données de test
    
    Returns:
    tuple: (X, y) - DataFrames sans features corrélées
    """
    print("\n🔍 Suppression des features corrélées...")
    
    correlated_columns = [
        'Total day minutes', 'Total eve minutes', 'Total night minutes', 
        'Total intl minutes', 'Voice mail plan'
    ]
    
    # Ne supprimer que les colonnes qui existent
    columns_to_drop = [col for col in correlated_columns if col in X.columns]
    
    X = X.drop(columns_to_drop, axis=1)
    y = y.drop(columns_to_drop, axis=1)
    
    print(f"✅ Features supprimées: {columns_to_drop}")
    
    return X, y

def prepare_training_data(X, y, sampling_strategy=0.3):
    """
    Prépare les données pour l'entraînement
    
    Parameters:
    X (DataFrame): Données d'entraînement
    y (DataFrame): Données de test
    sampling_strategy (float): Ratio pour la rééchantillonnage
    
    Returns:
    tuple: Données préparées pour l'entraînement
    """
    print("\n📚 Préparation des données pour l'entraînement...")
    
    # Séparation features/target
    X_train = X.drop(['Churn'], axis=1)
    y_train = X['Churn']
    X_test = y.drop(['Churn'], axis=1)
    y_test = y['Churn']
    
    print(f"   - X_train shape: {X_train.shape}")
    print(f"   - y_train shape: {y_train.shape}")
    print(f"   - Distribution initiale: {y_train.value_counts().to_dict()}")
    
    # Rééchantillonnage pour gérer le déséquilibre
    smote_enn = SMOTEENN(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    
    print(f"   - Après rééchantillonnage: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Données préparées avec succès")
    
    return X_train_scaled, X_test_scaled, y_resampled, y_test, scaler

def initialize_models():
    """
    Initialise tous les modèles à tester
    
    Returns:
    dict: Dictionnaire des modèles initialisés
    """
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42),
        "Support Vector Machine": SVC(class_weight='balanced', random_state=42, probability=True),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42),
        "Neural Network": MLPClassifier(random_state=42, max_iter=1000)
    }
    
    return models

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Entraîne et évalue tous les modèles
    
    Parameters:
    models (dict): Dictionnaire des modèles
    X_train (array): Features d'entraînement
    X_test (array): Features de test
    y_train (array): Target d'entraînement
    y_test (array): Target de test
    
    Returns:
    dict: Résultats de l'évaluation
    """
    print("\n🤖 Entraînement et évaluation des modèles...")
    print("=" * 60)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n🔧 Entraînement de {model_name}...")
        
        try:
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Évaluation
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            cr = classification_report(y_test, y_pred)
            
            # Métriques supplémentaires si disponibles
            metrics_dict = {
                'model': model,
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': cr
            }
            
            # Probabilities pour les métriques avancées
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics_dict['roc_auc'] = roc_auc_score(y_test, y_proba)
                metrics_dict['log_loss'] = log_loss(y_test, y_proba)
            
            results[model_name] = metrics_dict
            
            print(f"   ✅ {model_name} - Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"   ❌ Erreur avec {model_name}: {e}")
            results[model_name] = {'error': str(e)}
    
    return results

def optimize_random_forest(X_train, X_test, y_train, y_test, n_trials=50):
    """
    Optimise Random Forest avec Optuna
    
    Parameters:
    X_train (array): Features d'entraînement
    X_test (array): Features de test
    y_train (array): Target d'entraînement
    y_test (array): Target de test
    n_trials (int): Nombre d'essais d'optimisation
    
    Returns:
    dict: Meilleurs hyperparamètres
    """
    print(f"\n🎯 Optimisation de Random Forest ({n_trials} essais)...")
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 30, step=5)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"✅ Meilleurs paramètres RF: {study.best_params}")
    print(f"✅ Meilleure accuracy: {study.best_value:.4f}")
    
    return study.best_params

def optimize_xgboost(X_train, X_test, y_train, y_test, n_trials=50):
    """
    Optimise XGBoost avec Optuna
    
    Parameters:
    X_train (array): Features d'entraînement
    X_test (array): Features de test
    y_train (array): Target d'entraînement
    y_test (array): Target de test
    n_trials (int): Nombre d'essais d'optimisation
    
    Returns:
    dict: Meilleurs hyperparamètres
    """
    print(f"\n🎯 Optimisation de XGBoost ({n_trials} essais)...")
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        gamma = trial.suggest_float('gamma', 0, 5)
        
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"✅ Meilleurs paramètres XGBoost: {study.best_params}")
    print(f"✅ Meilleure accuracy: {study.best_value:.4f}")
    
    return study.best_params

def evaluate_model(model, X_test, y_test, model_name=""):
    """
    Évalue un modèle avec plusieurs métriques
    
    Parameters:
    model: Modèle entraîné
    X_test (array): Features de test
    y_test (array): Target de test
    model_name (str): Nom du modèle pour l'affichage
    
    Returns:
    dict: Métriques d'évaluation
    """
    print(f"\n📊 Évaluation détaillée de {model_name}")
    print("=" * 50)
    
    y_pred = model.predict(X_test)
    
    # Métriques de base
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    # Métriques avancées
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    metrics_dict = {
        'accuracy': accuracy,
        'kappa': kappa,
        'mcc': mcc,
        'confusion_matrix': cm,
        'classification_report': cr
    }
    
    # Métriques avec probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        logloss = log_loss(y_test, y_proba)
        metrics_dict.update({
            'roc_auc': roc_auc,
            'log_loss': logloss
        })
    
    # Affichage des résultats
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    if 'roc_auc' in metrics_dict:
        print(f"ROC AUC: {metrics_dict['roc_auc']:.4f}")
        print(f"Log Loss: {metrics_dict['log_loss']:.4f}")
    
    print(f"\nMatrice de confusion:")
    print(cm)
    print(f"\nRapport de classification:")
    print(cr)
    
    return metrics_dict

def plot_roc_curve(model, X_test, y_test, model_name=""):
    """
    Trace la courbe ROC
    
    Parameters:
    model: Modèle entraîné
    X_test (array): Features de test
    y_test (array): Target de test
    model_name (str): Nom du modèle
    """
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

# ...existing code...
def save_model(model, scaler, encoders, filepath):
    """
    Sauvegarde le modèle, le scaler et les encodeurs dans un seul fichier joblib/pka.
    Accepte les extensions: .joblib, .pkl, .pka (préférer .pka si demandé).
    """
    try:
        import os
        parent = os.path.dirname(filepath)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        # Autoriser .pka comme extension valide
        if not filepath.lower().endswith(('.joblib', '.pkl', '.pka')):
            filepath = filepath + '.joblib'

        bundle = {
            "model": model,
            "scaler": scaler,
            "encoders": encoders,
            "metadata": {
                "saved_at": pd.Timestamp.now(),
                "model_type": type(model).__name__
            }
        }

        # compression raisonnable
        joblib.dump(bundle, filepath, compress=3)
        print(f"✅ Modèle sauvegardé avec succès dans : {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du modèle : {e}")
        return None

def load_saved_model(filepath):
    """
    Recharge le modèle + scaler + encodeurs depuis un fichier joblib/pkl/pka.
    Retourne tuple (model, scaler, encoders) ou (None, None, None) en cas d'erreur.
    """
    try:
        import os
        candidates = [filepath]

        # si le chemin fourni n'existe pas, tenter avec extensions usuelles
        if not os.path.exists(filepath):
            for ext in ('.pka', '.joblib', '.pkl'):
                if os.path.exists(filepath + ext):
                    candidates = [filepath + ext]
                    break

        bundle = None
        tried = []
        for f in candidates:
            tried.append(f)
            try:
                bundle = joblib.load(f)
                filepath = f
                break
            except Exception:
                bundle = None

        if bundle is None:
            # essayer toutes les extensions si les candidats initiaux n'ont pas marché
            for ext in ('.pka', '.joblib', '.pkl'):
                path_with_ext = filepath if filepath.lower().endswith(ext) else filepath + ext
                if os.path.exists(path_with_ext):
                    try:
                        bundle = joblib.load(path_with_ext)
                        filepath = path_with_ext
                        break
                    except Exception:
                        bundle = None

        if bundle is None:
            print(f"❌ Impossible de charger le fichier. Fichiers testés: {tried}")
            return None, None, None

        print(f"✅ Modèle chargé avec succès depuis : {filepath}")
        return bundle.get("model"), bundle.get("scaler"), bundle.get("encoders")
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None, None, None
# ...existing code...

def load_model(filepath):
    """
    Charge le modèle sauvegardé (alias pour load_saved_model pour la compatibilité)
    
    Parameters:
    filepath (str): Chemin vers le fichier du modèle
    
    Returns:
    dict: Dictionnaire contenant le modèle, scaler et encodeurs
    """
    model, scaler, encoders = load_saved_model(filepath)
    if model is not None:
        return {
            'model': model,
            'scaler': scaler,
            'encoders': encoders
        }
    return None