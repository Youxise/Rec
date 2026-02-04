"""
Module de gestion de la base de données.

Ce module fournit une interface propre pour interagir avec PostgreSQL
en utilisant SQLAlchemy avec :
- Pool de connexions
- Context managers pour la gestion automatique des transactions
- Gestion d'erreurs robuste
- Logging des opérations

Exemple d'utilisation :
    from src.database import get_db, init_db
    
    # Initialiser la DB au démarrage
    init_db()
    
    # Utiliser dans une fonction
    with get_db() as db:
        result = db.execute("SELECT * FROM anime_metadata")
"""

import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, text, event, pool
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import pandas as pd
from config import Config

# Configuration du logging
logger = logging.getLogger(__name__)

# ==================================
# Engine SQLAlchemy (singleton)
# ==================================

_engine = None
_SessionFactory = None


def get_engine():
    """
    Retourne l'engine SQLAlchemy (singleton).
    
    L'engine est créé une seule fois et réutilisé.
    Il gère automatiquement un pool de connexions.
    
    Returns:
        Engine SQLAlchemy
    """
    global _engine
    
    if _engine is None:
        try:
            _engine = create_engine(
                Config.DATABASE_URL,
                # Pool de connexions
                poolclass=pool.QueuePool,
                pool_size=Config.SQLALCHEMY_POOL_SIZE,
                max_overflow=20,  # Connexions supplémentaires si besoin
                pool_recycle=Config.SQLALCHEMY_POOL_RECYCLE,
                pool_pre_ping=True,  # Vérifier la connexion avant utilisation
                pool_timeout=Config.SQLALCHEMY_POOL_TIMEOUT,
                # Options de performance
                echo=Config.DEBUG,  # Logger les requêtes SQL en mode debug
            )
            
            # Event listener pour logger les connexions
            @event.listens_for(_engine, "connect")
            def receive_connect(dbapi_conn, connection_record):
                logger.debug("Nouvelle connexion DB établie")
            
            logger.info(f"Engine SQLAlchemy créé avec succès. Pool size: {Config.SQLALCHEMY_POOL_SIZE}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'engine : {e}")
            raise
    
    return _engine


def get_session_factory():
    """
    Retourne la factory de sessions (singleton).
    
    Returns:
        Session factory
    """
    global _SessionFactory
    
    if _SessionFactory is None:
        engine = get_engine()
        session_factory = sessionmaker(bind=engine)
        _SessionFactory = scoped_session(session_factory)
        logger.info("Session factory créée")
    
    return _SessionFactory


@contextmanager
def get_db():
    """
    Context manager pour obtenir une session de base de données.
    
    Gère automatiquement :
    - Ouverture de la connexion
    - Commit en cas de succès
    - Rollback en cas d'erreur
    - Fermeture de la connexion
    
    Yields:
        Session SQLAlchemy
        
    Example:
        with get_db() as db:
            result = db.execute(text("SELECT * FROM anime_metadata"))
            for row in result:
                print(row)
    """
    SessionFactory = get_session_factory()
    session = SessionFactory()
    
    try:
        yield session
        session.commit()
        logger.debug("Transaction commitée avec succès")
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Erreur DB, rollback effectué : {e}")
        raise
    finally:
        session.close()
        logger.debug("Session fermée")


def get_connection():
    """
    Retourne une connexion brute (pour pandas ou requêtes simples).
    
    Returns:
        Connection SQLAlchemy
        
    Note:
        Préférez get_db() pour les opérations ORM.
        Cette fonction est utile pour pandas.read_sql().
    """
    engine = get_engine()
    return engine.connect()


# ==================================
# Fonctions utilitaires
# ==================================

def execute_query(query, params=None):
    """
    Exécute une requête SQL et retourne les résultats.
    
    Args:
        query (str): Requête SQL à exécuter
        params (dict, optional): Paramètres de la requête
        
    Returns:
        List[dict]: Résultats de la requête
        
    Example:
        results = execute_query(
            "SELECT * FROM anime_metadata WHERE MAL_ID = :id",
            params={"id": 1234}
        )
    """
    with get_db() as db:
        result = db.execute(text(query), params or {})
        return [dict(row._mapping) for row in result]


def fetch_dataframe(query, params=None):
    """
    Exécute une requête SQL et retourne un DataFrame pandas.
    
    Args:
        query (str): Requête SQL
        params (dict, optional): Paramètres de la requête
        
    Returns:
        pd.DataFrame: Résultats sous forme de DataFrame
        
    Example:
        df = fetch_dataframe("SELECT * FROM anime_metadata LIMIT 10")
    """
    try:
        engine = get_engine()
        df = pd.read_sql(query, engine, params=params)
        logger.debug(f"DataFrame chargé : {len(df)} lignes")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement du DataFrame : {e}")
        raise


def check_connection():
    """
    Vérifie que la connexion à la base de données fonctionne.
    
    Returns:
        bool: True si la connexion fonctionne, False sinon
    """
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        logger.info("Connexion DB vérifiée avec succès")
        return True
    except Exception as e:
        logger.error(f"Échec de la connexion DB : {e}")
        return False


def init_db():
    """
    Initialise la base de données.
    
    - Vérifie la connexion
    - Crée l'engine
    - Prépare les pools
    
    Raises:
        Exception: Si la connexion échoue
    """
    logger.info("Initialisation de la base de données...")
    
    try:
        engine = get_engine()
        
        # Vérifier la connexion
        if not check_connection():
            raise Exception("Impossible de se connecter à la base de données")
        
        logger.info("Base de données initialisée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la DB : {e}")
        raise


def close_db():
    """
    Ferme proprement toutes les connexions à la base de données.
    
    À appeler lors de l'arrêt de l'application.
    """
    global _engine, _SessionFactory
    
    try:
        if _SessionFactory is not None:
            _SessionFactory.remove()
            logger.info("Session factory fermée")
        
        if _engine is not None:
            _engine.dispose()
            logger.info("Engine et pool de connexions fermés")
            _engine = None
            _SessionFactory = None
            
    except Exception as e:
        logger.error(f"Erreur lors de la fermeture de la DB : {e}")


# ==================================
# Cache pour les données fréquentes
# ==================================

_cache = {}


def get_cached_dataframe(table_name, ttl=300):
    """
    Retourne un DataFrame avec cache.
    
    Args:
        table_name (str): Nom de la table
        ttl (int): Durée de vie du cache en secondes (5min par défaut)
        
    Returns:
        pd.DataFrame: Données de la table
    """
    import time
    
    cache_key = f"df_{table_name}"
    
    # Vérifier si le cache est valide
    if cache_key in _cache:
        cached_data, timestamp = _cache[cache_key]
        if time.time() - timestamp < ttl:
            logger.debug(f"Cache hit pour {table_name}")
            return cached_data.copy()
    
    # Charger depuis la DB
    logger.debug(f"Cache miss pour {table_name}, chargement depuis la DB")
    df = fetch_dataframe(f"SELECT * FROM {table_name}")
    _cache[cache_key] = (df, time.time())
    
    return df.copy()


def clear_cache():
    """Vide le cache des DataFrames."""
    global _cache
    _cache = {}
    logger.info("Cache vidé")
