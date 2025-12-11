#!/bin/bash
# Script pour choisir le service à lancer dans Docker

echo "Service demandé: $SERVICE"

if [ "$SERVICE" = "fastapi" ]; then
    echo "Lancement FastAPI..."
    uvicorn app:app --host 0.0.0.0 --port 8000
elif [ "$SERVICE" = "streamlit" ]; then
    echo "Lancement Streamlit..."
    streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
elif [ "$SERVICE" = "test" ]; then
    echo "Lancement des tests..."
    make test
else
    echo "Aucun service défini, accès au shell"
    /bin/bash
fi
