FROM python:3.9-slim

WORKDIR /app

# OpenCV-Abhängigkeiten
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Python-Pakete installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere die Anwendung
COPY . .

# Startbefehl - kann je nach Bedarf angepasst werden
# Trainingsmodus: python hand_detection.py --mode train --epochs 50
# Erkennungsmodus: python hand_detection.py --mode detect
ENTRYPOINT ["python", "hand_detection.py"]
CMD ["--help"]
