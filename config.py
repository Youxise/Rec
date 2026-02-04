"""
Configuration centralisée pour l'application Rec.

Ce module charge toutes les variables d'environnement et fournit
une configuration type-safe pour l'ensemble de l'application.

Avantages :
- Sécurité : pas de credentials hardcodés
- Flexibilité : facile à modifier selon l'environnement (dev/prod)
- Maintenabilité : configuration centralisée en un seul endroit
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
# override=False garantit que les variables d'environnement système (Docker) ont priorité
load_dotenv(override=False)

class Config:
    """Configuration de base pour l'application."""
    
    # Chemins
    BASE_DIR = Path(__file__).parent
    DATA_PATH = BASE_DIR / os.getenv("DATA_PATH", "data")
    EMBEDDINGS_PATH = BASE_DIR / os.getenv("EMBEDDINGS_PATH", "src/embeddings.npy")
    LOG_PATH = BASE_DIR / "logs"
    
    # Base de données
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "rec_db")
    
    # Construction de l'URL de la base de données
    # Format: postgresql://user:password@host:port/database
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # SQLAlchemy configuration
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_POOL_SIZE = 10  # Nombre de connexions dans le pool
    SQLALCHEMY_POOL_RECYCLE = 3600  # Recycler les connexions après 1h
    SQLALCHEMY_POOL_TIMEOUT = 30  # Timeout de 30 secondes
    
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    # Recommandations
    CONTENT_WEIGHT = float(os.getenv("CONTENT_WEIGHT", "0.75"))
    COLLABORATIVE_WEIGHT = float(os.getenv("COLLABORATIVE_WEIGHT", "0.25"))
    TOP_N_RECOMMENDATIONS = int(os.getenv("TOP_N_RECOMMENDATIONS", "10"))
    MAX_SELECTED_ANIMES = 5  # Nombre max d'animes sélectionnables
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    
    @classmethod
    def validate(cls):
        """
        Valide que toutes les configurations critiques sont présentes.
        
        Raises:
            ValueError: Si une configuration critique est manquante
        """
        required_vars = {
            "DB_USER": cls.DB_USER,
            "DB_PASSWORD": cls.DB_PASSWORD,
            "DB_NAME": cls.DB_NAME,
            "SECRET_KEY": cls.SECRET_KEY
        }
        
        missing = [key for key, value in required_vars.items() 
                   if not value or value.startswith("your_") or value.endswith("_here_change_in_production")]
        
        if missing and cls.FLASK_ENV == "production":
            raise ValueError(
                f"Configuration critique manquante ou invalide en production : {', '.join(missing)}. "
                f"Veuillez configurer ces variables dans votre fichier .env"
            )
    
    @classmethod
    def init_app(cls, app):
        """
        Initialise l'application Flask avec la configuration.
        
        Args:
            app: Instance Flask
        """
        # Créer les dossiers nécessaires
        cls.LOG_PATH.mkdir(exist_ok=True)
        cls.DATA_PATH.mkdir(exist_ok=True)
        
        # Valider la configuration
        cls.validate()


class DevelopmentConfig(Config):
    """Configuration pour le développement."""
    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Configuration pour les tests."""
    TESTING = True
    DEBUG = True
    # Utiliser une base de données de test
    DB_NAME = "rec_db_test"
    DATABASE_URL = f"postgresql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{DB_NAME}"
    SQLALCHEMY_DATABASE_URI = DATABASE_URL


class ProductionConfig(Config):
    """Configuration pour la production."""
    DEBUG = False
    TESTING = False
    # En production, toutes les variables doivent être définies
    
    @classmethod
    def validate(cls):
        """Validation stricte pour la production."""
        super().validate()
        
        # Vérifications supplémentaires pour la production
        if cls.SECRET_KEY == "dev-secret-key-change-in-production":
            raise ValueError("SECRET_KEY doit être changée en production!")
        
        if len(cls.SECRET_KEY) < 32:
            raise ValueError("SECRET_KEY doit faire au moins 32 caractères en production!")


# Dictionnaire de configurations
config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig
}


def get_config(env=None):
    """
    Retourne la configuration appropriée selon l'environnement.
    
    Args:
        env: Nom de l'environnement (development, testing, production)
        
    Returns:
        Classe de configuration appropriée
    """
    if env is None:
        env = os.getenv("FLASK_ENV", "development")
    
    return config.get(env, config["default"])
