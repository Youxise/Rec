"""
Module de configuration du logging pour l'application Rec.

Fournit un système de logging structuré avec :
- Rotation automatique des fichiers
- Différents niveaux de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Formatage coloré pour la console
- Logs JSON pour la production

Exemple d'utilisation :
    from src.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Application démarrée")
    logger.error("Une erreur est survenue", exc_info=True)
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from config import Config

# Couleurs ANSI pour la console
class LogColors:
    """Codes couleur ANSI pour le terminal."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'


class ColoredFormatter(logging.Formatter):
    """
    Formatter personnalisé avec couleurs pour la console.
    
    Colore les logs selon leur niveau :
    - DEBUG: Cyan
    - INFO: Green
    - WARNING: Yellow
    - ERROR: Red
    - CRITICAL: Red Bold
    """
    
    COLORS = {
        'DEBUG': LogColors.CYAN,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.RED + LogColors.BOLD,
    }
    
    def format(self, record):
        """Formate le record avec des couleurs."""
        # Sauvegarder le levelname original
        levelname = record.levelname
        
        # Ajouter la couleur
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{LogColors.RESET}"
        
        # Formater le message
        formatted = super().format(record)
        
        # Restaurer le levelname original
        record.levelname = levelname
        
        return formatted


def setup_logging():
    """
    Configure le système de logging pour toute l'application.
    
    Crée :
    - Un handler pour la console (coloré, niveau INFO+)
    - Un handler pour les fichiers (rotation automatique, tous les niveaux)
    - Un handler pour les erreurs critiques (fichier séparé)
    """
    # Créer le dossier de logs s'il n'existe pas
    log_dir = Path(Config.LOG_PATH)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Configuration de base
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Supprimer les handlers existants pour éviter les duplications
    root_logger.handlers.clear()
    
    # ==================================
    # Handler 1 : Console (coloré)
    # ==================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    console_format = ColoredFormatter(
        '%(levelname)s | %(asctime)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # ==================================
    # Handler 2 : Fichier principal (rotation)
    # ==================================
    # Rotation : 10 MB max par fichier, garde 5 anciens fichiers
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / Config.LOG_FILE.split('/')[-1],
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    file_format = logging.Formatter(
        '%(levelname)s | %(asctime)s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)
    
    # ==================================
    # Handler 3 : Erreurs critiques (fichier séparé)
    # ==================================
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'errors.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    root_logger.addHandler(error_handler)
    
    # ==================================
    # Réduire le bruit des librairies tierces
    # ==================================
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    
    # Log initial
    root_logger.info("=" * 80)
    root_logger.info(f"Logging configuré - Niveau: {Config.LOG_LEVEL}")
    root_logger.info(f"Logs sauvegardés dans: {log_dir.absolute()}")
    root_logger.info("=" * 80)


def get_logger(name):
    """
    Retourne un logger configuré pour un module.
    
    Args:
        name (str): Nom du module (généralement __name__)
        
    Returns:
        logging.Logger: Logger configuré
        
    Example:
        logger = get_logger(__name__)
        logger.info("Module chargé")
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Adapter pour ajouter du contexte aux logs.
    
    Permet d'ajouter automatiquement des informations
    comme l'utilisateur, l'ID de session, etc.
    
    Example:
        logger = LoggerAdapter(get_logger(__name__), {'user_id': 123})
        logger.info("Action effectuée")
        # Output: INFO | ... | user_id=123 | Action effectuée
    """
    
    def process(self, msg, kwargs):
        """Ajoute le contexte au message."""
        # Convertir le contexte en string
        context = ' | '.join(f"{k}={v}" for k, v in self.extra.items())
        return f"{context} | {msg}" if context else msg, kwargs


def log_function_call(func):
    """
    Décorateur pour logger automatiquement les appels de fonction.
    
    Logs :
    - Entrée dans la fonction avec les arguments
    - Sortie de la fonction avec le résultat
    - Exceptions éventuelles
    
    Example:
        @log_function_call
        def calculate_score(anime_id):
            return score
    """
    import functools
    
    logger = get_logger(func.__module__)
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Log de l'entrée
        logger.debug(f"Appel de {func.__name__}() avec args={args}, kwargs={kwargs}")
        
        try:
            # Exécuter la fonction
            result = func(*args, **kwargs)
            
            # Log de la sortie
            logger.debug(f"{func.__name__}() terminé avec succès")
            
            return result
            
        except Exception as e:
            # Log de l'erreur
            logger.error(f"Erreur dans {func.__name__}() : {e}", exc_info=True)
            raise
    
    return wrapper


def log_performance(threshold_ms=1000):
    """
    Décorateur pour logger les performances des fonctions.
    
    Alerte si une fonction prend plus de temps que le seuil.
    
    Args:
        threshold_ms (int): Seuil en millisecondes
        
    Example:
        @log_performance(threshold_ms=500)
        def slow_function():
            # Si prend > 500ms, un warning sera loggé
            pass
    """
    import functools
    import time
    
    def decorator(func):
        logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000
            
            if duration_ms > threshold_ms:
                logger.warning(
                    f"{func.__name__}() a pris {duration_ms:.2f}ms "
                    f"(seuil: {threshold_ms}ms)"
                )
            else:
                logger.debug(f"{func.__name__}() exécuté en {duration_ms:.2f}ms")
            
            return result
        
        return wrapper
    
    return decorator
