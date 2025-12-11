# main.py
from pathlib import Path
from flask import Flask
from model_pipeline import (
    load_data, explore_data, preprocess_data, handle_outliers, create_features,
    remove_correlated_features, prepare_training_data, initialize_models,
    train_and_evaluate_models, save_model, load_saved_model, evaluate_model, plot_roc_curve
)

# Définir les chemins
PROJECT_DIR = Path(__file__).parent  # dossier du script main.py
TRAIN_CSV = PROJECT_DIR / "churn-bigml-80.csv"
TEST_CSV = PROJECT_DIR / "churn-bigml-20.csv"
RESULT_DIR = PROJECT_DIR / "resultat"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = RESULT_DIR / "best_model.pka"

def main():
    # 1) Chargement des données
    X, y = load_data(str(TRAIN_CSV), str(TEST_CSV))
    if X is None or y is None:
        return

    # 2) Exploration rapide
    explore_data(X, y)

    # 3) Prétraitement
    X_proc, y_proc, encoders = preprocess_data(X, y)

    # 4) Gestion des outliers (uniquement sur X)
    X_proc = handle_outliers(X_proc)
    X_proc = X_proc.reset_index(drop=True)
    y_proc = y_proc.reset_index(drop=True)

    # 5) Création de nouvelles features
    X_proc, y_proc = create_features(X_proc, y_proc)

    # 6) Suppression des features corrélées
    X_proc, y_proc = remove_correlated_features(X_proc, y_proc)

    # 7) Préparation des données (scaling + rééchantillonnage)
    X_train, X_test, y_train, y_test, scaler = prepare_training_data(X_proc, y_proc, sampling_strategy=0.3)

    # 8) Initialisation des modèles
    models = initialize_models()

    # 9) Entraînement et évaluation des modèles
    results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)

    # 10) Sélection du meilleur modèle selon l'accuracy
    best_name, best_score = None, -1
    for name, res in results.items():
        if isinstance(res, dict) and 'error' not in res:
            score = res.get('accuracy', -1.0)
            if score > best_score:
                best_score = score
                best_name = name

    if best_name is None:
        print("❌ Aucun modèle entraîné avec succès.")
        return

    best_model = results[best_name]['model']
    print(f"\n🏆 Meilleur modèle: {best_name} (accuracy={best_score:.4f})")

    # 11) Sauvegarde du modèle
    saved_path = save_model(best_model, scaler, encoders, str(MODEL_PATH))
    if saved_path is None:
        print("❌ Échec de la sauvegarde du modèle.")
        return

    # 12) Chargement du modèle sauvegardé
    model_loaded, scaler_loaded, encoders_loaded = load_saved_model(saved_path)
    if model_loaded is None:
        print("❌ Échec du chargement du modèle sauvegardé.")
        return

    # 13) Évaluation détaillée du modèle chargé
    evaluate_model(model_loaded, X_test, y_test, model_name=f"{best_name} (loaded)")

    # 14) Tracer la ROC curve
    try:
        plot_roc_curve(model_loaded, X_test, y_test, model_name=best_name)
    except Exception:
        pass

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello Asma, Flask is running!"

if __name__ == "__main__":
    main()

# model_pipeline.py