"""
Application Flask pour le système de recommandation d'animes Rec.

Ce module contient l'application principale avec toutes les routes API.
Utilise un système hybride de recommandations :
- Content-based (embeddings + features)
- Collaborative filtering (FP-Growth)

Routes:
    GET  / : Page d'accueil
    GET  /search_animes : Recherche d'animes
    POST /recommendations : Génération de recommandations
    GET  /discover_animes : Découverte d'animes avec filtres
    GET  /health : Health check pour monitoring
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import BadRequest
import sys
import os
from pathlib import Path

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Imports locaux
from config import get_config, Config
from src.database import init_db, close_db, get_cached_dataframe, check_connection
from src.logger import setup_logging, get_logger
from src.models import ContentRecommender, FPGrowthRecommender, combine_results

# Configuration
env = os.getenv("FLASK_ENV", "development")
app_config = get_config(env)

# Initialisation Flask
app = Flask(__name__)
app.config.from_object(app_config)

# Initialisation du logging
setup_logging()
logger = get_logger(__name__)

# Sécurité : CORS et Rate Limiting
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Désactiver rate limiting si pytest est en cours d'exécution
import sys
is_testing = 'pytest' in sys.modules or app.config.get('TESTING', False)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    enabled=not is_testing
)

# ============================================
# Initialisation des modèles (singleton)
# ============================================

logger.info("=" * 80)
logger.info(f"Démarrage de l'application Rec (env={env})")
logger.info("=" * 80)

logger.info("=" * 80)
logger.info(f"Démarrage de l'application Rec (env={env})")
logger.info("=" * 80)

# Initialiser la base de données
try:
    init_db()
    logger.info("✓ Base de données initialisée")
except Exception as e:
    logger.error(f"✗ Erreur lors de l'initialisation de la DB : {e}")
    sys.exit(1)

# Charger les recommenders (singleton pattern)
logger.info("Chargement des modèles de recommandation...")
recommender_content = ContentRecommender()
recommender_collaborative = FPGrowthRecommender()
logger.info("✓ Modèles chargés (lazy loading activé)")

# Cache des métadonnées d'animes
anime_metadata = None


def get_anime_metadata():
    """
    Retourne les métadonnées d'animes avec cache.
    
    Returns:
        pd.DataFrame: Métadonnées des animes
    """
    global anime_metadata
    if anime_metadata is None:
        anime_metadata = get_cached_dataframe("anime_metadata", ttl=3600)
        logger.info(f"✓ Métadonnées chargées : {len(anime_metadata)} animes")
    return anime_metadata


# ============================================
# Routes
# ============================================

@app.route('/health')
def health():
    """
    Health check endpoint pour monitoring.
    
    Vérifie :
    - Application en ligne
    - Connexion à la base de données
    - Modèles chargés
    
    Returns:
        JSON avec statut de santé
    """
    status = {
        "status": "healthy",
        "environment": env,
        "database": "connected" if check_connection() else "disconnected",
        "models": "loaded"
    }
    
    status_code = 200 if status["database"] == "connected" else 503
    
    return jsonify(status), status_code
    return jsonify(status), status_code


@app.route('/')
def index():
    """Page d'accueil de l'application."""
    logger.info("Page d'accueil chargée")
    return render_template("index.html")


@app.route('/search_animes', methods=['GET'])
@limiter.limit("30 per minute")
def search_animes():
    """
    Recherche d'animes par titre (autocomplétion).
    
    Query params:
        query (str): Terme de recherche
        
    Returns:
        JSON: Liste d'animes correspondants avec titre et année
        
    Example:
        GET /search_animes?query=naruto
        Response: [{"Title": "Naruto", "Release": 2002}, ...]
    """
    query = request.args.get('query', '').strip()
    
    if not query:
        return jsonify([])
    
    if len(query) < 2:
        return jsonify({"error": "La recherche doit contenir au moins 2 caractères"}), 400
    
    logger.debug(f"Recherche : '{query}'")
    
    try:
        metadata = get_anime_metadata()
        
        # Recherche case-insensitive
        results = metadata[
            metadata['Title'].str.contains(query, case=False, na=False)
        ]
        
        # Limiter à 10 résultats pour les performances
        results = results.head(10)
        
        response = results[['Title', 'Release']].to_dict(orient='records')
        
        logger.debug(f"✓ {len(response)} résultats trouvés pour '{query}'")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche : {e}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500

        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route('/recommendations', methods=['POST'])
@limiter.limit("10 per minute")
def recommendations():
    """
    Génère des recommandations basées sur une sélection d'animes.
    
    Request body (JSON):
        {
            "titles": ["Naruto", "One Piece"],
            "top_n": 10  // optionnel
        }
        
    Returns:
        JSON: Liste des animes recommandés avec toutes leurs métadonnées
        
    Errors:
        400: Données invalides
        404: Anime(s) non trouvé(s)
        500: Erreur serveur
    """
    try:
        # Validation des données d'entrée
        data = request.get_json()
        
        if not data:
            logger.warning("Requête sans données JSON")
            return jsonify({"error": "Corps de requête invalide"}), 400
        
        if 'titles' not in data:
            logger.warning("Champ 'titles' manquant")
            return jsonify({"error": "Le champ 'titles' est requis"}), 400
        
        anime_titles = data['titles']
        top_n = data.get('top_n', Config.TOP_N_RECOMMENDATIONS)
        
        # Validation
        if not isinstance(anime_titles, list):
            return jsonify({"error": "'titles' doit être une liste"}), 400
        
        if len(anime_titles) == 0:
            return jsonify({"error": "Au moins un anime doit être sélectionné"}), 400
        
        if len(anime_titles) > Config.MAX_SELECTED_ANIMES:
            return jsonify({
                "error": f"Maximum {Config.MAX_SELECTED_ANIMES} animes autorisés"
            }), 400
        
        logger.info(f"Génération de recommandations pour : {anime_titles}")
        
        # Récupérer les métadonnées
        metadata = get_anime_metadata()
        
        # Vérifier que tous les titres existent
        missing_titles = [
            title for title in anime_titles 
            if title not in metadata["Title"].values
        ]
        
        if missing_titles:
            logger.warning(f"Titres non trouvés : {missing_titles}")
            return jsonify({
                "error": f"Animes non trouvés : {', '.join(missing_titles)}"
            }), 404
        
        # Obtenir les IDs MyAnimeList
        anime_ids = metadata.loc[
            metadata["Title"].isin(anime_titles), 
            "MAL_ID"
        ].tolist()
        
        logger.info(f"IDs : {anime_ids}")
        
        # Générer les recommandations
        logger.info("Application des modèles de recommandation...")
        
        # Modèle 1 : Content-based
        similar_anime_ids_content = recommender_content.get_recommendations(
            anime_ids, 
            top_n=top_n
        )
        
        # Modèle 2 : Collaborative filtering
        similar_anime_ids_collab = recommender_collaborative.get_recommendations(
            anime_ids
        )
        
        # Combiner les résultats
        final_recommend_list = combine_results(
            similar_anime_ids_collab,
            similar_anime_ids_content,
            metadata,
            k=top_n
        )
        
        logger.info(f"✓ {len(final_recommend_list)} recommandations finales générées")
        
        # Récupérer les détails des animes recommandés
        recommended_animes = metadata[
            metadata["MAL_ID"].isin(final_recommend_list)
        ]
        
        response = recommended_animes.to_dict(orient='records')
        
        logger.info(f"✓ Recommandations envoyées avec succès")
        return jsonify(response), 200

    except ValueError as e:
        logger.warning(f"Erreur de validation : {e}")
        return jsonify({"error": str(e)}), 400
    
    except BadRequest as e:
        logger.warning(f"Requête invalide : {e}")
        return jsonify({"error": "Corps de requête JSON invalide"}), 400
        
    except Exception as e:
        logger.error(f"Erreur lors de la génération de recommandations : {e}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route('/discover_animes', methods=['GET'])
@limiter.limit("30 per minute")
def discover_animes():
    """
    Découvre des animes avec filtres et tri.
    
    Query params:
        sort_by (str): Colonne de tri (Score, Popularity, Release). Défaut: Score
        order (str): Ordre (asc, desc). Défaut: desc
        year (int): Filtrer par année de sortie
        limit (int): Nombre de résultats (max 50). Défaut: 10
        
    Returns:
        JSON: Liste d'animes filtrés et triés
        
    Example:
        GET /discover_animes?sort_by=Score&year=2020&limit=20
    """
    try:
        # Paramètres de requête
        sort_by = request.args.get('sort_by', 'Score')
        order = request.args.get('order', 'desc').lower()
        year = request.args.get('year', type=int)
        limit = min(request.args.get('limit', type=int, default=10), 50)
        
        # Validation
        valid_sort_columns = ['Score', 'Popularity', 'Release', 'Episodes']
        if sort_by not in valid_sort_columns:
            return jsonify({
                "error": f"sort_by invalide. Valeurs autorisées : {', '.join(valid_sort_columns)}"
            }), 400
        
        if order not in ['asc', 'desc']:
            return jsonify({"error": "order doit être 'asc' ou 'desc'"}), 400
        
        logger.info(f"Découverte : sort_by={sort_by}, order={order}, year={year}, limit={limit}")
        
        # Charger les données
        metadata = get_anime_metadata()
        
        # Filtrer par année si spécifié
        if year:
            metadata = metadata[metadata['Release'] == year]
            logger.debug(f"{len(metadata)} animes trouvés pour l'année {year}")
        
        # Trier
        ascending = (order == 'asc')
        results = metadata.sort_values(by=sort_by, ascending=ascending).head(limit)
        
        response = results.to_dict(orient='records')
        
        logger.info(f"✓ {len(response)} animes retournés")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Erreur lors de la découverte : {e}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500


# ============================================
# Gestion des erreurs
# ============================================

@app.errorhandler(404)
def not_found(error):
    """Gestion des erreurs 404."""
    logger.warning(f"404 Not Found : {request.url}")
    return jsonify({"error": "Route non trouvée"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Gestion des erreurs 500."""
    logger.error(f"500 Internal Server Error : {error}", exc_info=True)
    return jsonify({"error": "Erreur interne du serveur"}), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    """Gestion du rate limiting."""
    logger.warning(f"Rate limit dépassé : {request.remote_addr}")
    return jsonify({"error": "Trop de requêtes. Veuillez réessayer plus tard."}), 429


# ============================================
# Cleanup au shutdown
# ============================================

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Ferme proprement les connexions à la base de données."""
    if exception:
        logger.error(f"Exception lors du teardown : {exception}")


def cleanup():
    """Nettoyage à l'arrêt de l'application."""
    logger.info("Arrêt de l'application...")
    close_db()
    logger.info("✓ Connexions fermées proprement")


import atexit
atexit.register(cleanup)


# ============================================
# Point d'entrée
# ============================================

# ============================================
# Point d'entrée
# ============================================

if __name__ == '__main__':
    """
    Démarre l'application Flask.
    
    En production, utiliser plutôt un serveur WSGI comme Gunicorn :
        gunicorn -w 4 -b 0.0.0.0:5000 app:app
    """
    logger.info("Démarrage du serveur Flask...")
    logger.info(f"Mode debug : {app.config['DEBUG']}")
    logger.info(f"Environnement : {env}")
    
    # Configuration du serveur
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    # En production, debug doit être False
    app.run(
        host=host,
        port=port,
        debug=app.config['DEBUG'],
        use_reloader=app.config['DEBUG']
    )
