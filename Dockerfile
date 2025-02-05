# Utilisation d'une image Python comme base
FROM python:3.10

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers du projet dans le conteneur
COPY . .

# Installer les dépendances
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exposer le port (exemple avec Flask)
EXPOSE 5000

# Commande pour démarrer l'application
CMD ["python", "app.py"]  # Remplace `app.py` par ton fichier principal
