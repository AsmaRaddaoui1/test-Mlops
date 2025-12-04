import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Churn Client",
    page_icon="📊",
    layout="wide"
)

# Titre de l'application
st.title("📊 Dashboard de Prédiction de Churn Client")
st.markdown("Interface pour interagir avec l'API de prédiction de désabonnement")

# URL de l'API
API_URL = "http://localhost:8000"

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisir une page:", ["Prédiction", "Réentraînement", "Statut API"])

# Fonction pour vérifier le statut de l'API
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200, response.json()
    except:
        return False, {}

# Page de prédiction
if page == "Prédiction":
    st.header("🎯 Prédire le Churn d'un Client")
    
    # Vérifier le statut de l'API
    api_ok, health_data = check_api_status()
    
    if not api_ok:
        st.error("❌ L'API n'est pas accessible. Vérifiez qu'elle est démarrée sur localhost:8000")
        st.stop()
    
    st.success("✅ API connectée avec succès")
    
    # Formulaire en deux colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Informations générales")
        state = st.selectbox("État", ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", 
                                    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
                                    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                                    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
                                    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"])
        
        account_length = st.number_input("Durée du compte (jours)", min_value=0, max_value=300, value=100)
        area_code = st.selectbox("Code régional", ["408", "415", "510"])
        
        international_plan = st.radio("Plan international", ["No", "Yes"])
        voice_mail_plan = st.radio("Messagerie vocale", ["No", "Yes"])
        number_vmail_messages = st.number_input("Messages vocaux", min_value=0, max_value=100, value=0)
    
    with col2:
        st.subheader("Statistiques d'appels")
        
        # Appels de jour
        total_day_minutes = st.number_input("Minutes jour", min_value=0.0, max_value=400.0, value=200.0)
        total_day_calls = st.number_input("Appels jour", min_value=0, max_value=200, value=100)
        total_day_charge = st.number_input("Coût jour ($)", min_value=0.0, max_value=100.0, value=34.0)
        
        # Appels de soir
        total_eve_minutes = st.number_input("Minutes soir", min_value=0.0, max_value=400.0, value=200.0)
        total_eve_calls = st.number_input("Appels soir", min_value=0, max_value=200, value=100)
        total_eve_charge = st.number_input("Coût soir ($)", min_value=0.0, max_value=100.0, value=17.0)
        
        # Appels de nuit
        total_night_minutes = st.number_input("Minutes nuit", min_value=0.0, max_value=400.0, value=200.0)
        total_night_calls = st.number_input("Appels nuit", min_value=0, max_value=200, value=100)
        total_night_charge = st.number_input("Coût nuit ($)", min_value=0.0, max_value=100.0, value=9.0)
        
        # Appels internationaux
        total_intl_minutes = st.number_input("Minutes internationales", min_value=0.0, max_value=30.0, value=10.0)
        total_intl_calls = st.number_input("Appels internationaux", min_value=0, max_value=20, value=4)
        total_intl_charge = st.number_input("Coût international ($)", min_value=0.0, max_value=10.0, value=2.7)
        
        customer_service_calls = st.slider("Appels service client", min_value=0, max_value=10, value=1)
    
    # Bouton de prédiction
    if st.button("🔮 Prédire le Churn", type="primary"):
        # Préparer les données
        data = {
            "State": state,
            "Account_length": account_length,
            "Area_code": area_code,
            "International_plan": international_plan,
            "Voice_mail_plan": voice_mail_plan,
            "Number_vmail_messages": number_vmail_messages,
            "Total_day_minutes": total_day_minutes,
            "Total_day_calls": total_day_calls,
            "Total_day_charge": total_day_charge,
            "Total_eve_minutes": total_eve_minutes,
            "Total_eve_calls": total_eve_calls,
            "Total_eve_charge": total_eve_charge,
            "Total_night_minutes": total_night_minutes,
            "Total_night_calls": total_night_calls,
            "Total_night_charge": total_night_charge,
            "Total_intl_minutes": total_intl_minutes,
            "Total_intl_calls": total_intl_calls,
            "Total_intl_charge": total_intl_charge,
            "Customer_service_calls": customer_service_calls
        }
        
        try:
            # Appel à l'API
            response = requests.post(f"{API_URL}/predict", json=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Affichage des résultats
                st.success("✅ Prédiction effectuée avec succès!")
                
                # Métriques
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result["prediction"] == 1:
                        st.error("🚨 RISQUE ÉLEVÉ de Churn")
                    else:
                        st.success("✅ FAIBLE RISQUE de Churn")
                
                with col2:
                    st.metric("Probabilité de Churn", f"{result['probability']:.2%}")
                
                with col3:
                    st.metric("Prédiction", "OUI" if result["prediction"] == 1 else "NON")
                
                # Graphique de probabilité
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode = "gauge+number+delta",
                    value = result["probability"] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Score de Risque de Churn"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Facteurs de risque
                st.subheader("🔍 Analyse des facteurs de risque")
                
                risk_factors = []
                if international_plan == "Yes":
                    risk_factors.append("📞 Plan international activé")
                if customer_service_calls >= 4:
                    risk_factors.append(f"📞 {customer_service_calls} appels service client")
                if total_day_charge > 50:
                    risk_factors.append(f"💸 Coût élevé jour: ${total_day_charge}")
                if total_day_minutes > 300:
                    risk_factors.append(f"⏱️ Minutes jour élevées: {total_day_minutes}")
                
                if risk_factors:
                    st.warning("Facteurs de risque détectés:")
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.info("Aucun facteur de risque majeur détecté")
                    
            else:
                st.error(f"❌ Erreur API: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Erreur de connexion: {e}")

# Page de réentraînement
elif page == "Réentraînement":
    st.header("🔄 Réentraînement du Modèle")
    
    st.info("""
    Cette fonctionnalité permet de réentraîner le modèle avec de nouvelles données.
    **Format requis:** Les données doivent inclure la colonne 'Churn' (True/False)
    """)
    
    # Exemple de données
    st.subheader("📋 Exemple de format de données")
    example_data = [
        {
            "State": "TX", "Account_length": 67, "Area_code": "415",
            "International_plan": "Yes", "Voice_mail_plan": "No",
            "Number_vmail_messages": 0, "Total_day_minutes": 320.5,
            "Total_day_calls": 55, "Total_day_charge": 54.49,
            "Total_eve_minutes": 290.1, "Total_eve_calls": 45,
            "Total_eve_charge": 24.66, "Total_night_minutes": 280.2,
            "Total_night_calls": 35, "Total_night_charge": 12.61,
            "Total_intl_minutes": 15.3, "Total_intl_calls": 7,
            "Total_intl_charge": 4.13, "Customer_service_calls": 5,
            "Churn": True
        }
    ]
    
    st.json(example_data)
    
    # Upload de fichier ou saisie manuelle
    st.subheader("📤 Charger de nouvelles données")
    
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            new_data = pd.read_csv(uploaded_file)
            st.write("Aperçu des données chargées:")
            st.dataframe(new_data.head())
            
            # Convertir en format JSON pour l'API
            data_for_api = new_data.to_dict('records')
            
            # Hyperparamètres
            st.subheader("⚙️ Hyperparamètres")
            col1, col2 = st.columns(2)
            
            with col1:
                n_estimators = st.number_input("n_estimators", min_value=10, max_value=500, value=100)
                max_depth = st.number_input("max_depth", min_value=3, max_value=50, value=10)
            
            with col2:
                test_size = st.slider("test_size", min_value=0.1, max_value=0.5, value=0.2)
                sampling_strategy = st.slider("sampling_strategy", min_value=0.1, max_value=1.0, value=0.3)
            
            if st.button("🚀 Lancer le réentraînement", type="primary"):
                retrain_data = {
                    "new_data": data_for_api,
                    "hyperparameters": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "random_state": 42
                    },
                    "test_size": test_size,
                    "sampling_strategy": sampling_strategy
                }
                
                try:
                    with st.spinner("Réentraînement en cours..."):
                        response = requests.post(f"{API_URL}/retrain", json=retrain_data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Réentraînement terminé avec succès!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Accuracy", f"{result['model_accuracy']:.2%}")
                        with col2:
                            st.metric("Échantillons", result['new_samples'])
                        with col3:
                            st.metric("Sauvegardé", "✅" if result['model_saved'] else "❌")
                        
                        st.info(result['message'])
                    else:
                        st.error(f"❌ Erreur: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
                    
        except Exception as e:
            st.error(f"❌ Erreur de lecture du fichier: {e}")

# Page statut API
elif page == "Statut API":
    st.header("📊 Statut de l'API")
    
    if st.button("🔄 Vérifier le statut"):
        api_ok, health_data = check_api_status()
        
        if api_ok:
            st.success("✅ API en ligne et fonctionnelle")
            st.json(health_data)
        else:
            st.error("❌ API hors ligne")
    
    # Statistiques d'utilisation
    st.subheader("📈 Informations techniques")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Port API", "8000")
    
    with col2:
        st.metric("Protocole", "HTTP")
    
    with col3:
        st.metric("Documentation", "Swagger UI")

# Footer
st.markdown("---")
st.markdown("Développé avec Streamlit & FastAPI • [Documentation API](http://localhost:8000/docs)")