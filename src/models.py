"""
Modèles de recommandation pour l'application Rec.

Ce module contient les algorithmes de recommandation :
- ContentRecommender : Recommandations basées sur le contenu (embeddings + features)
- FPGrowthRecommender : Recommandations collaboratives (association rules)
- combine_results : Hybridation des deux approches

Optimisations :
- Chargement lazy des données et modèles
- Cache des embeddings et règles d'association
- Vectorisation des calculs
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import fpgrowth, association_rules
from pathlib import Path

from src.database import fetch_dataframe, get_cached_dataframe, get_engine
from src.logger import get_logger, log_performance
from config import Config

logger = get_logger(__name__)

# ------------------------------------------------- Content Filtering Recommender ------------------------------------------------- #

class ContentRecommender:
    """
    Système de recommandation basé sur le contenu.
    
    Utilise :
    - Embeddings textuels (synopsis, genres, thèmes) via Sentence Transformers
    - Features numériques (score, popularité, nombre d'épisodes)
    - Similarité cosinus pondérée
    
    Attributes:
        data (pd.DataFrame): Métadonnées des animes
        text_features (np.ndarray): Embeddings textuels
        numeric_features (np.ndarray): Features numériques normalisées
    """
    
    def __init__(self, embeddings_path=None):
        """
        Initialise le modèle basé sur le contenu.
        
        Args:
            embeddings_path (str, optional): Chemin vers le fichier d'embeddings
        """
        logger.info("Initialisation du ContentRecommender...")
        
        # Lazy loading : on charge les données seulement quand nécessaire
        self._data = None
        self._text_features = None
        self._numeric_features = None
        self._combined_features = None
        
        # Chemin des embeddings
        self.embeddings_path = embeddings_path or Config.EMBEDDINGS_PATH
        
        logger.info("ContentRecommender initialisé (lazy loading activé)")
    
    @property
    def data(self):
        """Charge les données d'anime (lazy loading)."""
        if self._data is None:
            logger.info("Chargement des métadonnées d'animes...")
            self._data = get_cached_dataframe("anime_metadata", ttl=3600)
            logger.info(f"✓ {len(self._data)} animes chargés")
        return self._data
    
    @property
    def text_features(self):
        """Charge les embeddings textuels (lazy loading)."""
        if self._text_features is None:
            logger.info(f"Chargement des embeddings depuis {self.embeddings_path}...")
            self._text_features = np.load(self.embeddings_path)
            logger.info(f"✓ Embeddings chargés : {self._text_features.shape}")
        return self._text_features
    
    @property
    def numeric_features(self):
        """Charge les features numériques (lazy loading)."""
        if self._numeric_features is None:
            logger.info("Chargement des features numériques...")
            df_pp = get_cached_dataframe("anime_metadata_pp", ttl=3600)
            self._numeric_features = df_pp[["Episodes", "Popularity", "Score", "Demographic"]].fillna(0).values
            logger.info(f"✓ Features numériques chargées : {self._numeric_features.shape}")
        return self._numeric_features
    
    @log_performance(threshold_ms=100)
    def get_combined_features(self):
        """
        Combine les embeddings textuels avec les features numériques.
        
        Returns:
            np.ndarray: Features combinées
        """
        if self._combined_features is None:
            logger.debug("Combinaison des features...")
            self._combined_features = np.hstack([self.text_features, self.numeric_features])
        return self._combined_features
    
    @log_performance(threshold_ms=500)
    def get_recommendations(self, anime_ids, top_n=None, weights=None):
        """
        Recommande des animes similaires pour plusieurs titres en entrée.
        
        Args:
            anime_ids (int or list): ID(s) d'anime(s) MyAnimeList
            top_n (int, optional): Nombre de recommandations. Défaut: Config.TOP_N_RECOMMENDATIONS
            weights (tuple, optional): Poids (texte, numérique). Défaut: Config weights
            
        Returns:
            list: Liste des MAL_ID recommandés
            
        Raises:
            ValueError: Si un anime_id n'existe pas
            
        Example:
            recommender = ContentRecommender()
            recommendations = recommender.get_recommendations([1, 5, 10], top_n=5)
        """
        # Paramètres par défaut
        top_n = top_n or Config.TOP_N_RECOMMENDATIONS
        weights = weights or (Config.CONTENT_WEIGHT, Config.COLLABORATIVE_WEIGHT)
        
        # Normaliser l'entrée
        if isinstance(anime_ids, int):
            anime_ids = [anime_ids]
        
        logger.info(f"Génération de {top_n} recommandations pour {len(anime_ids)} anime(s)")
        
        # Vérifier que les animes existent
        data = self.data
        anime_indices = data[data["MAL_ID"].isin(anime_ids)].index
        
        if len(anime_indices) == 0:
            raise ValueError(f"Aucun anime trouvé pour les IDs : {anime_ids}")
        
        if len(anime_indices) < len(anime_ids):
            missing = set(anime_ids) - set(data.loc[anime_indices, "MAL_ID"].tolist())
            logger.warning(f"Animes non trouvés : {missing}")
        
        # Combiner les caractéristiques
        combined_features = self.get_combined_features()
        text_dim = self.text_features.shape[1]
        
        # Calculer la similarité pour chaque anime d'entrée
        logger.debug("Calcul de la similarité cosinus...")
        text_similarities = cosine_similarity(
            combined_features[anime_indices, :text_dim],
            combined_features[:, :text_dim]
        )
        numeric_similarities = cosine_similarity(
            combined_features[anime_indices, text_dim:],
            combined_features[:, text_dim:]
        )
        
        # Combinaison pondérée des similarités
        total_similarities = (
            weights[0] * text_similarities + weights[1] * numeric_similarities
        )
        
        # Moyenne des similarités pour tous les animes d'entrée
        aggregated_similarity = total_similarities.mean(axis=0)
        
        # Obtenir les indices des animes les plus similaires
        # Exclure les animes d'entrée
        similar_indices = aggregated_similarity.argsort()[-(top_n + len(anime_ids)):][::-1]
        similar_indices = [idx for idx in similar_indices if idx not in anime_indices][:top_n]
        
        # Retourner les MAL_ID
        recommended_ids_list = data.iloc[similar_indices]["MAL_ID"].tolist()
        
        logger.info(f"✓ {len(recommended_ids_list)} recommandations générées")
        return recommended_ids_list

# ------------------------------------------------- Collaborative Recommender ------------------------------------------------- #

class FPGrowthRecommender:
    """
    Système de recommandation collaborative basé sur les règles d'association.
    
    Utilise l'algorithme FP-Growth pour découvrir des patterns
    dans les habitudes de visionnage des utilisateurs.
    
    Si l'utilisateur a aimé l'anime A, et que beaucoup d'utilisateurs
    qui ont aimé A ont aussi aimé B, alors B sera recommandé.
    
    Attributes:
        ratings (pd.DataFrame): Données de notations des utilisateurs
        rules (pd.DataFrame): Règles d'association pré-calculées
    """
    
    def __init__(self):
        """Initialise le système de recommandation collaborative."""
        logger.info("Initialisation du FPGrowthRecommender...")
        
        # Lazy loading
        self._ratings = None
        self._rules = None
        
        logger.info("FPGrowthRecommender initialisé (lazy loading activé)")
    
    @property
    def ratings(self):
        """Charge les ratings (lazy loading)."""
        if self._ratings is None:
            logger.info("Chargement des ratings...")
            self._ratings = get_cached_dataframe("anime_transactions", ttl=3600)
            logger.info(f"✓ {len(self._ratings)} ratings chargés")
        return self._ratings
    
    @property
    def rules(self):
        """Charge et nettoie les règles d'association (lazy loading)."""
        if self._rules is None:
            logger.info("Chargement des règles d'association...")
            rules = get_cached_dataframe("association_rules", ttl=3600)
            
            # Reconversion et filtrage
            rules['antecedents'] = rules['antecedents'].apply(lambda x: set(eval(x)))
            rules['consequents'] = rules['consequents'].apply(lambda x: set(eval(x)))
            
            # Filtrer pour garder seulement les règles simples (1→1)
            rules = rules[rules['antecedents'].apply(lambda x: len(x) == 1)]
            rules = rules[rules['consequents'].apply(lambda x: len(x) == 1)]
            
            self._rules = rules
            logger.info(f"✓ {len(self._rules)} règles d'association chargées")
        
        return self._rules
    
    @log_performance(threshold_ms=200)
    def get_recommendations(self, anime_ids):
        """
        Recommande des animes basés sur les règles d'association.
        
        Args:
            anime_ids (int or list): ID(s) d'anime(s) MyAnimeList
            
        Returns:
            list: Liste des MAL_ID recommandés, triés par support
            
        Example:
            recommender = FPGrowthRecommender()
            recommendations = recommender.get_recommendations([1, 5, 10])
        """
        # Normaliser l'entrée
        if isinstance(anime_ids, int):
            anime_ids = [anime_ids]
        
        logger.info(f"Recherche de recommandations collaboratives pour {len(anime_ids)} anime(s)")
        
        rules = self.rules
        similar_index = []
        
        # Pour chaque anime d'entrée, trouver les règles correspondantes
        for anime_id in anime_ids:
            for _, row in rules.iterrows():
                # Vérifier si l'anime est dans les antécédents
                if anime_id in row['antecedents']:
                    similar_index.append({
                        'consequents': row['consequents'],
                        'support': row['support']
                    })
        
        recommended_index_list = []
        
        if similar_index:
            # Convertir en DataFrame
            similar_index_df = pd.DataFrame(similar_index)
            
            # Trier par support (confiance)
            similar_index_df = similar_index_df.sort_values(by="support", ascending=False)
            
            # Extraire les IDs recommandés
            similar_index = list(similar_index_df["consequents"])
            recommended_index_list = [next(iter(fset)) for fset in similar_index if fset]
            
            # Si plusieurs animes en entrée, garder seulement les duplicatas
            # (recommandés pour plusieurs animes d'entrée = plus pertinents)
            if len(anime_ids) > 1:
                recommended_index_list = find_duplicates(recommended_index_list)
        
        logger.info(f"✓ {len(recommended_index_list)} recommandations collaboratives générées")
        return recommended_index_list
    
    def _rebuild_rules(self):
        """
        Reconstruit les règles d'association à partir des ratings.
        
        ATTENTION : Fonction coûteuse en temps et ressources.
        À utiliser uniquement quand les données de ratings changent significativement.
        
        Les règles sont sauvegardées dans la base de données.
        """
        logger.warning("Reconstruction des règles d'association (opération longue)...")
        
        # Créer une matrice utilisateur-item binaire
        user_item_matrix = self.ratings.pivot_table(
            index='user_id',
            columns='MAL_ID',
            values='rating',
            aggfunc='count',
            fill_value=0
        )
        
        # Convertir en binaire (1 si interaction, 0 sinon)
        user_item_matrix = user_item_matrix.map(lambda x: 1 if x > 0 else 0)
        
        logger.info(f"Matrice utilisateur-item : {user_item_matrix.shape}")
        
        # Appliquer FP-Growth
        logger.info("Application de FP-Growth...")
        frequent_itemsets = fpgrowth(user_item_matrix, min_support=0.001, use_colnames=True)
        
        # Éliminer les singletons
        frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 1)]
        
        logger.info(f"✓ {len(frequent_itemsets)} itemsets fréquents trouvés")
        
        # Générer les règles d'association
        if not frequent_itemsets.empty:
            logger.info("Génération des règles d'association...")
            rules = association_rules(
                frequent_itemsets,
                metric="lift",
                min_threshold=0,
                num_itemsets=len(frequent_itemsets),
                support_only=True
            )
            rules = rules[['antecedents', 'consequents', 'support']]
            
            # Convertir les frozensets en strings pour la sauvegarde
            rules['antecedents'] = rules['antecedents'].apply(lambda x: str(x))
            rules['consequents'] = rules['consequents'].apply(lambda x: str(x))
            
            # Sauvegarder dans la DB
            engine = get_engine()
            rules.to_sql('association_rules', engine, if_exists='replace', index=False)
            
            logger.info(f"✓ {len(rules)} règles d'association sauvegardées dans la DB")
        else:
            logger.warning("Aucun itemset fréquent trouvé, aucune règle générée")


# ------------------------------------------------- Utility Functions ------------------------------------------------- #

def find_duplicates(input_list):
    """
    Trouve les éléments en double dans une liste.
    
    Utile pour identifier les recommandations communes
    à plusieurs animes d'entrée (plus pertinentes).
    
    Args:
        input_list (list): Liste avec potentiellement des doublons
        
    Returns:
        list: Liste des éléments qui apparaissent au moins 2 fois
        
    Example:
        >>> find_duplicates([1, 2, 3, 2, 4, 3])
        [2, 3]
    """
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)


@log_performance(threshold_ms=100)
def combine_results(list1, list2, data, k=None):
    """
    Combine les résultats de deux systèmes de recommandation.
    
    Stratégie hybride :
    - Prend 50% de recommandations collaboratives (list1)
    - Prend 50% de recommandations basées sur le contenu (list2)
    - Évite les doublons
    
    Args:
        list1 (list): Recommandations collaboratives (FP-Growth)
        list2 (list): Recommandations de contenu
        data (pd.DataFrame): Métadonnées des animes (pour validation)
        k (int, optional): Nombre total de recommandations. Défaut: Config.TOP_N_RECOMMENDATIONS
        
    Returns:
        list: Liste combinée de k MAL_IDs
        
    Example:
        combined = combine_results(
            collaborative_recs, 
            content_recs, 
            anime_metadata, 
            k=10
        )
    """
    k = k or Config.TOP_N_RECOMMENDATIONS
    
    unique_from_list1 = []
    unique_from_list2 = []
    seen = set()
    
    # Prendre k/2 éléments uniques de list1 (FP-Growth)
    for item in list1:
        if item not in seen:
            unique_from_list1.append(item)
            seen.add(item)
        if len(unique_from_list1) == k // 2:
            break

    # Prendre les éléments restants de list2 (Content)
    # Vérifier qu'ils existent dans la base
    valid_ids = set(data["MAL_ID"].tolist())
    
    for item in list2:
        if item not in seen and item in valid_ids:
            unique_from_list2.append(item)
            seen.add(item)
        if len(unique_from_list2) == k - len(unique_from_list1):
            break

    # Combiner les résultats
    result = unique_from_list1 + unique_from_list2
    
    logger.debug(
        f"Combinaison : {len(unique_from_list1)} collaboratives + "
        f"{len(unique_from_list2)} content = {len(result)} total"
    )
    
    return result


# ------------------------------------------------- Script d'entraînement ------------------------------------------------- #

if __name__ == "__main__":
    """
    Script pour reconstruire les règles d'association.
    
    ATTENTION : Opération longue et coûteuse.
    À exécuter uniquement quand les données de ratings changent significativement.
    
    Usage:
        python -m src.models
    """
    from src.logger import setup_logging
    
    setup_logging()
    logger.info("Démarrage de la reconstruction des règles d'association...")
    
    try:
        recommender = FPGrowthRecommender()
        recommender._rebuild_rules()
        logger.info("✓ Reconstruction terminée avec succès")
    except Exception as e:
        logger.error(f"✗ Erreur lors de la reconstruction : {e}", exc_info=True)

