# Używamy lekkiego obrazu Pythona
FROM python:3.12-slim

# Ustawiamy katalog roboczy w kontenerze
WORKDIR /app

# Kopiujemy pliki projektu
COPY train_model.py /app
COPY train_model_linear.py /app
COPY server.py /app
COPY checkModel.py /app
COPY checkModel_linear.py /app

# Instalacja bibliotek Pythona
RUN pip install --no-cache-dir numpy pandas requests joblib scikit-learn flask matplotlib

# Domyślne uruchomienie serwera Flask
CMD ["python3", "server.py"]