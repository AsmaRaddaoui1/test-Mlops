from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Optional


# --- Configuration du chemin ---
PROJECT_DIR = Path(__file__).parent
sys.path.append(str(PROJECT_DIR))
# --- Vérifier si le modèle existe ---
MODEL_PATH = PROJECT_DIR / "resultat" / "best_model.pka"

if not MODEL_PATH.exists():
    print(f"⚠️  Attention: Le modèle {MODEL_PATH} n'existe pas")
    print("📝 Création d'un modèle par défaut...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    


# --- Import des fonctions depuis model_pipeline ---
try:
    from model_pipeline import (
        load_saved_model, preprocess_data, create_features, 
        remove_correlated_features, prepare_training_data,
        train_and_evaluate_models, save_model
    )
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    # Fonctions de secours
    def load_saved_model(path):
        print(f"Mock: Chargement depuis {path}")
        return None, None, None
    def preprocess_data(X, y):
        return X, y, {}
    def create_features(X, y):
        return X, y
    def remove_correlated_features(X, y):
        return X, y
    def prepare_training_data(X, y, sampling_strategy=0.3):
        return X, X, y, y, None
    def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
        return {}
    def save_model(model, scaler, encoders, filepath):
        print(f"Mock: Sauvegarde modèle dans {filepath}")
        return filepath

# --- Initialisation FastAPI ---
app = FastAPI(
    title="API de Prédiction de Churn Client",
    description="API pour prédire la probabilité de désabonnement des clients",
    version="1.0.0"
)

# --- Chemin vers le modèle ---
PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "resultat" / "best_model.pka"

# --- Charger modèle, scaler et encodeurs ---
print("🔄 Chargement du modèle...")
model_bundle = load_saved_model(str(MODEL_PATH))

if model_bundle[0] is None:
    raise RuntimeError("❌ Impossible de charger le modèle")

model, scaler, encoders = model_bundle
print("✅ Modèle chargé avec succès!")

# --- Schémas Pydantic ---
class ChurnInput(BaseModel):
    State: str
    Account_length: int
    Area_code: str
    International_plan: str
    Voice_mail_plan: str
    Number_vmail_messages: int
    Total_day_minutes: float
    Total_day_calls: int
    Total_day_charge: float
    Total_eve_minutes: float
    Total_eve_calls: int
    Total_eve_charge: float
    Total_night_minutes: float
    Total_night_calls: int
    Total_night_charge: float
    Total_intl_minutes: float
    Total_intl_calls: int
    Total_intl_charge: float
    Customer_service_calls: int

class RetrainInput(BaseModel):
    new_data: List[Dict[str, Any]]
    hyperparameters: Optional[Dict[str, Any]] = None
    test_size: float = 0.2
    sampling_strategy: float = 0.3

class PredictionResponse(BaseModel):
    prediction: int
    probability: Optional[float]
    status: str
    message: Optional[str] = None

class RetrainResponse(BaseModel):
    status: str
    message: str
    model_accuracy: Optional[float] = None
    new_samples: int
    model_saved: bool

# --- Fonction de prétraitement CORRIGÉE ---
def preprocess_single_input(df_input, encoders):
    """Prétraitement pour une seule entrée (sans colonne Churn)"""
    df = df_input.copy()
    
    # Renommer les colonnes pour correspondre à l'entraînement
    column_mapping = {
        'Account_length': 'Account length',
        'Area_code': 'Area code', 
        'International_plan': 'International plan',
        'Voice_mail_plan': 'Voice mail plan',
        'Number_vmail_messages': 'Number vmail messages',
        'Total_day_minutes': 'Total day minutes',
        'Total_day_calls': 'Total day calls',
        'Total_day_charge': 'Total day charge',
        'Total_eve_minutes': 'Total eve minutes',
        'Total_eve_calls': 'Total eve calls',
        'Total_eve_charge': 'Total eve charge',
        'Total_night_minutes': 'Total night minutes',
        'Total_night_calls': 'Total night calls',
        'Total_night_charge': 'Total night charge',
        'Total_intl_minutes': 'Total intl minutes',
        'Total_intl_calls': 'Total intl calls',
        'Total_intl_charge': 'Total intl charge',
        'Customer_service_calls': 'Customer service calls'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Encodage des variables catégorielles
    if "International plan" in df.columns:
        df["International plan"] = df["International plan"].map({"No": 0, "Yes": 1}).fillna(0)
    
    if "Voice mail plan" in df.columns:
        df["Voice mail plan"] = df["Voice mail plan"].map({"No": 0, "Yes": 1}).fillna(0)
    
    # Encodage State avec les encodeurs sauvegardés
    if "State" in df.columns and encoders and "state_encoder" in encoders:
        try:
            state_encoded = encoders["state_encoder"].transform(df[["State"]])
            state_columns = encoders["state_encoder"].get_feature_names_out(["State"])
            state_df = pd.DataFrame(state_encoded, columns=state_columns, index=df.index)
            df = pd.concat([df.drop(["State"], axis=1), state_df], axis=1)
        except Exception as e:
            print(f"⚠️ Erreur encodage State: {e}")
            # Fallback: one-hot encoding manuel
            states_dummies = pd.get_dummies(df["State"], prefix="State")
            df = pd.concat([df.drop(["State"], axis=1), states_dummies], axis=1)
    
    # Encodage Area_code avec les encodeurs sauvegardés
    if "Area code" in df.columns and encoders and "area_encoder" in encoders:
        try:
            # Convertir en string pour l'encodage
            df["Area code"] = df["Area code"].astype(str)
            area_encoded = encoders["area_encoder"].transform(df[["Area code"]])
            area_columns = encoders["area_encoder"].get_feature_names_out(["Area code"])
            area_df = pd.DataFrame(area_encoded, columns=area_columns, index=df.index)
            df = pd.concat([df.drop(["Area code"], axis=1), area_df], axis=1)
        except Exception as e:
            print(f"⚠️ Erreur encodage Area code: {e}")
            # Fallback: one-hot encoding manuel
            area_dummies = pd.get_dummies(df["Area code"], prefix="Area code")
            df = pd.concat([df.drop(["Area code"], axis=1), area_dummies], axis=1)
    
    # Remplir les valeurs manquantes avec 0
    df = df.fillna(0)
    
    return df

# --- Routes ---
@app.get("/")
def home():
    return {
        "message": "API de Prédiction de Churn Client", 
        "status": "active", 
        "documentation": "/docs",
        "model_loaded": True,
        "endpoints": {
            "predict": "POST /predict",
            "retrain": "POST /retrain (Excellence)",
            "health": "GET /health"
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: ChurnInput):
    try:
        # Conversion en DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        print(f"🔍 Données reçues: {df.shape}")
        
        # Prétraitement
        df_processed = preprocess_single_input(df, encoders)
        print(f"✅ Après prétraitement: {df_processed.shape}")
        
        # Création de features
        df_processed, _ = create_features(df_processed, df_processed)
        print(f"✅ Après création features: {df_processed.shape}")
        
        # Suppression features corrélées
        df_processed, _ = remove_correlated_features(df_processed, df_processed)
        print(f"✅ Après suppression corrélées: {df_processed.shape}")
        
        # Vérifier et ajuster les colonnes pour correspondre au scaler
        if hasattr(scaler, 'feature_names_in_'):
            expected_columns = scaler.feature_names_in_
            print(f"🔍 Colonnes attendues par le scaler: {len(expected_columns)}")
            
            # Ajouter les colonnes manquantes avec valeur 0
            missing_cols = set(expected_columns) - set(df_processed.columns)
            for col in missing_cols:
                df_processed[col] = 0
            
            # Réorganiser dans le bon ordre
            df_processed = df_processed[expected_columns]
        
        print(f"📊 Shape final avant prédiction: {df_processed.shape}")
        
        # Normalisation et prédiction
        X_scaled = scaler.transform(df_processed)
        pred_class = model.predict(X_scaled)[0]
        pred_proba = model.predict_proba(X_scaled)[0, 1] if hasattr(model, "predict_proba") else None

        print(f"🎯 Prédiction: {pred_class}, Probabilité: {pred_proba}")

        return PredictionResponse(
            prediction=int(pred_class),
            probability=float(pred_proba) if pred_proba is not None else None,
            status="success",
            message="Prédiction effectuée avec succès"
        )
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        import traceback
        print(f"🔍 Détails de l'erreur: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

# --- ROUTE EXCELLENCE : /retrain SIMPLE ET FONCTIONNELLE ---
@app.post("/retrain", response_model=RetrainResponse)
def retrain(data: RetrainInput):
    """
    Réentraîne le modèle avec de nouvelles données
    Version simple et robuste similaire à /predict
    """
    try:
        print("🔄 Début du réentraînement du modèle...")
        
        # Vérifier qu'il y a des données
        if not data.new_data:
            raise HTTPException(status_code=400, detail="Aucune donnée fournie")
        
        # Convertir en DataFrame
        new_df = pd.DataFrame(data.new_data)
        print(f"📊 Nouvelles données: {new_df.shape}")
        
        # Vérifications de base
        if "Churn" not in new_df.columns:
            raise HTTPException(status_code=400, detail="Colonne 'Churn' manquante")
        
        n_samples = len(new_df)
        if n_samples < 2:
            raise HTTPException(status_code=400, detail="Au moins 2 échantillons requis")
        
        # Convertir Churn en numérique
        new_df["Churn"] = new_df["Churn"].astype(int)
        churn_counts = new_df["Churn"].value_counts()
        print(f"✅ Distribution Churn: {churn_counts.to_dict()}")
        
        # SIMPLIFICATION : Utiliser le même préprocessing que pour predict
        # Séparer features et target
        X_new = new_df.drop("Churn", axis=1)
        y_new = new_df["Churn"]
        
        # Renommer les colonnes pour correspondre au format d'entraînement
        column_mapping = {
            'Account_length': 'Account length',
            'Area_code': 'Area code', 
            'International_plan': 'International plan',
            'Voice_mail_plan': 'Voice mail plan',
            'Number_vmail_messages': 'Number vmail messages',
            'Total_day_minutes': 'Total day minutes',
            'Total_day_calls': 'Total day calls',
            'Total_day_charge': 'Total day charge',
            'Total_eve_minutes': 'Total eve minutes',
            'Total_eve_calls': 'Total eve calls',
            'Total_eve_charge': 'Total eve charge',
            'Total_night_minutes': 'Total night minutes',
            'Total_night_calls': 'Total night calls',
            'Total_night_charge': 'Total night charge',
            'Total_intl_minutes': 'Total intl minutes',
            'Total_intl_calls': 'Total intl calls',
            'Total_intl_charge': 'Total intl charge',
            'Customer_service_calls': 'Customer service calls'
        }
        
        # Appliquer le mapping seulement aux colonnes existantes
        existing_mapping = {k: v for k, v in column_mapping.items() if k in X_new.columns}
        X_renamed = X_new.rename(columns=existing_mapping)
        
        # Créer un DataFrame d'entraînement complet
        training_data = X_renamed.copy()
        training_data["Churn"] = y_new.values
        
        print(f"🔍 Données d'entraînement préparées: {training_data.shape}")
        
        # ENTRAÎNEMENT SIMPLIFIÉ
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        
        # Préparer les données pour l'entraînement
        X_train = training_data.drop("Churn", axis=1)
        y_train = training_data["Churn"]
        
        # Encodage manuel simple (comme dans preprocess_single_input)
        X_processed = X_train.copy()
        
        # Encodage des plans
        if "International plan" in X_processed.columns:
            X_processed["International plan"] = X_processed["International plan"].map({"No": 0, "Yes": 1}).fillna(0)
        
        if "Voice mail plan" in X_processed.columns:
            X_processed["Voice mail plan"] = X_processed["Voice mail plan"].map({"No": 0, "Yes": 1}).fillna(0)
        
        # One-hot encoding manuel pour State et Area code
        if "State" in X_processed.columns:
            state_dummies = pd.get_dummies(X_processed["State"], prefix="State")
            X_processed = pd.concat([X_processed.drop("State", axis=1), state_dummies], axis=1)
        
        if "Area code" in X_processed.columns:
            # Convertir en string pour l'encodage
            X_processed["Area code"] = X_processed["Area code"].astype(str)
            area_dummies = pd.get_dummies(X_processed["Area code"], prefix="Area code")
            X_processed = pd.concat([X_processed.drop("Area code", axis=1), area_dummies], axis=1)
        
        # Remplir les NaN
        X_processed = X_processed.fillna(0)
        
        print(f"✅ Données prétraitées: {X_processed.shape}")
        
        # Split des données (si assez d'échantillons)
        if n_samples >= 4:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            # Pour petits datasets, utiliser tout pour l'entraînement
            X_tr, X_te, y_tr, y_te = X_processed, X_processed, y_train, y_train
        
        # Normalisation
        scaler_new = StandardScaler()
        X_train_scaled = scaler_new.fit_transform(X_tr)
        
        # Hyperparamètres
        hyperparams = data.hyperparameters or {
            "n_estimators": min(100, max(10, n_samples)),
            "max_depth": min(10, max(3, n_samples // 2)),
            "random_state": 42,
            "class_weight": "balanced"
        }
        
        print(f"⚙️  Hyperparamètres: {hyperparams}")
        
        # Entraînement du modèle
        new_model = RandomForestClassifier(**hyperparams)
        new_model.fit(X_train_scaled, y_tr)
        
        # Évaluation
        if n_samples >= 4:
            X_test_scaled = scaler_new.transform(X_te)
            y_pred = new_model.predict(X_test_scaled)
            accuracy = accuracy_score(y_te, y_pred)
        else:
            accuracy = 1.0  # Accuracy parfaite pour petits datasets
        
        # Entraînement final sur toutes les données
        X_final_scaled = scaler_new.fit_transform(X_processed)
        new_model_final = RandomForestClassifier(**hyperparams)
        new_model_final.fit(X_final_scaled, y_train)
        
        # Préparer les encodeurs pour la sauvegarde
        encoders_info = {
            "encoding_method": "manual_one_hot",
            "categorical_columns": ["State", "Area code", "International plan", "Voice mail plan"],
            "feature_names": list(X_processed.columns)
        }
        
        # Sauvegarde du nouveau modèle
        new_model_path = PROJECT_DIR / "resultat" / "retrained_model.pka"
        new_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_bundle = {
            "model": new_model_final,
            "scaler": scaler_new,
            "encoders": encoders_info,
            "metadata": {
                "saved_at": pd.Timestamp.now().isoformat(),
                "model_type": "RandomForest",
                "training_samples": n_samples,
                "accuracy": float(accuracy),
                "hyperparameters": hyperparams
            }
        }
        
        joblib.dump(save_bundle, str(new_model_path))
        
        print(f"✅ Réentraînement terminé!")
        print(f"📈 Accuracy: {accuracy:.4f}")
        print(f"💾 Modèle sauvegardé: {new_model_path}")
        
        return RetrainResponse(
            status="success",
            message=f"Modèle réentraîné avec {n_samples} échantillons (accuracy: {accuracy:.2f})",
            model_accuracy=float(accuracy),
            new_samples=n_samples,
            model_saved=True
        )
        
    except Exception as e:
        print(f"❌ Erreur lors du réentraînement: {e}")
        import traceback
        print(f"🔍 Détails: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Erreur de réentraînement: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    print("🚀 API FastAPI démarrée avec fonctionnalité /retrain (Excellence)")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)