#!/usr/bin/env python3
"""
Script principal pour l'exécution du pipeline de prédiction de churn
Usage: python main.py [--train_path PATH] [--test_path PATH] [--optimize] [--save_model]
"""

import argparse
import sys
import os

# Ajouter le chemin pour importer model_pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_pipeline import (
    load_data, explore_data, preprocess_data, handle_outliers,
    create_features, remove_correlated_features, prepare_training_data,
    initialize_models, train_and_evaluate_models, optimize_random_forest,
    optimize_xgboost, evaluate_model, plot_roc_curve, save_model
)

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Pipeline de prédiction de churn client')
    parser.add_argument('--train_path', type=str, default='churn-bigml-80.csv',
                       help='Chemin vers le fichier d\'entraînement')
    parser.add_argument('--test_path', type=str, default='churn-bigml-20.csv',
                       help='Chemin vers le fichier de test')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimiser les hyperparamètres')
    parser.add_argument('--save_model', action='store_true',
                       help='Sauvegarder le meilleur modèle')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Nombre d\'essais pour l\'optimisation')
    
    args = parser.parse_args()
    
    print("🚀 Démarrage du pipeline de prédiction de churn")
    print("=" * 60)
    
    # Étape 1: Chargement des données
    X, y = load_data(args.train_path, args.test_path)
    if X is None or y is None:
        sys.exit(1)
    
    # Étape 2: Exploration
    X, y = explore_data(X, y)
    
    # Étape 3: Prétraitement
    X, y, encoders = preprocess_data(X, y)
    
    # Étape 4: Gestion des outliers
    X = handle_outliers(X)
    
    # Étape 5: Feature engineering
    X, y = create_features(X, y)
    X, y = remove_correlated_features(X, y)
    
    # Étape 6: Préparation des données
    X_train_scaled, X_test_scaled, y_resampled, y_test, scaler = prepare_training_data(X, y)
    
    # Étape 7: Entraînement des modèles de base
    models = initialize_models()
    results = train_and_evaluate_models(models, X_train_scaled, X_test_scaled, y_resampled, y_test)
    
    # Afficher les résultats des modèles de base
    print("\n" + "=" * 60)
    print("📈 RÉSULTATS DES MODÈLES DE BASE")
    print("=" * 60)
    
    for model_name, result in results.items():
        if 'accuracy' in result:
            print(f"{model_name:25} | Accuracy: {result['accuracy']:.4f} | ", end="")
            if 'roc_auc' in result:
                print(f"ROC AUC: {result['roc_auc']:.4f}")
            else:
                print()
    
    # Étape 8: Optimisation (optionnelle)
    if args.optimize:
        print("\n" + "=" * 60)
        print("🎯 OPTIMISATION DES HYPERPARAMÈTRES")
        print("=" * 60)
        
        # Optimisation Random Forest
        best_rf_params = optimize_random_forest(
            X_train_scaled, X_test_scaled, y_resampled, y_test, args.n_trials
        )
        
        # Optimisation XGBoost
        best_xgb_params = optimize_xgboost(
            X_train_scaled, X_test_scaled, y_resampled, y_test, args.n_trials
        )
        
        # Entraînement des modèles optimisés
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        
        best_rf_model = RandomForestClassifier(**best_rf_params, random_state=42)
        best_rf_model.fit(X_train_scaled, y_resampled)
        
        best_xgb_model = XGBClassifier(**best_xgb_params, random_state=42)
        best_xgb_model.fit(X_train_scaled, y_resampled)
        
        # Évaluation des modèles optimisés
        print("\n" + "=" * 60)
        print("📊 ÉVALUATION DES MODÈLES OPTIMISÉS")
        print("=" * 60)
        
        rf_metrics = evaluate_model(best_rf_model, X_test_scaled, y_test, "Random Forest Optimisé")
        xgb_metrics = evaluate_model(best_xgb_model, X_test_scaled, y_test, "XGBoost Optimisé")
        
        # Courbes ROC
        plot_roc_curve(best_rf_model, X_test_scaled, y_test, "Random Forest Optimisé")
        plot_roc_curve(best_xgb_model, X_test_scaled, y_test, "XGBoost Optimisé")
        
        # Sauvegarde du meilleur modèle
        if args.save_model:
            # Choisir le meilleur modèle basé sur l'accuracy
            if rf_metrics['accuracy'] >= xgb_metrics['accuracy']:
                best_model = best_rf_model
                print("💾 Sauvegarde du modèle Random Forest")
            else:
                best_model = best_xgb_model
                print("💾 Sauvegarde du modèle XGBoost")
            
            save_model(best_model, scaler, encoders, 'best_churn_model.joblib')
    
    print("\n✅ Pipeline terminé avec succès!")

if __name__ == "__main__":
    main()
