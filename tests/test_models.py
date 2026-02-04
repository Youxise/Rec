"""
Tests pour les modèles de recommandation.

Ces tests vérifient le bon fonctionnement des recommenders
et des fonctions utilitaires.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from src.models import ContentRecommender, FPGrowthRecommender, combine_results, find_duplicates


class TestContentRecommender:
    """Tests pour le ContentRecommender."""
    
    @patch('src.models.get_cached_dataframe')
    @patch('src.models.np.load')
    def test_initialization(self, mock_np_load, mock_get_df):
        """Test que le recommender s'initialise correctement."""
        recommender = ContentRecommender()
        assert recommender._data is None  # Lazy loading
        assert recommender._text_features is None
        assert recommender._numeric_features is None
    
    @patch('src.models.get_cached_dataframe')
    @patch('src.models.np.load')
    def test_lazy_loading_data(self, mock_np_load, mock_get_df):
        """Test que les données sont chargées à la demande."""
        mock_df = pd.DataFrame({
            'MAL_ID': [1, 2, 3],
            'Title': ['Anime1', 'Anime2', 'Anime3']
        })
        mock_get_df.return_value = mock_df
        
        recommender = ContentRecommender()
        
        # Première accès : doit charger
        data = recommender.data
        assert len(data) == 3
        mock_get_df.assert_called_once()
        
        # Deuxième accès : doit utiliser le cache
        data2 = recommender.data
        assert len(data2) == 3
        mock_get_df.assert_called_once()  # Pas d'appel supplémentaire
    
    @patch('src.models.get_cached_dataframe')
    @patch('src.models.np.load')
    def test_get_recommendations_single_id(self, mock_np_load, mock_get_df):
        """Test recommandations pour un seul anime."""
        # Mock des données
        mock_df = pd.DataFrame({
            'MAL_ID': [1, 2, 3, 4, 5],
            'Title': ['A', 'B', 'C', 'D', 'E']
        })
        mock_get_df.return_value = mock_df
        
        # Mock des embeddings
        mock_np_load.return_value = np.random.rand(5, 384)
        
        # Mock des features numériques
        mock_numeric = pd.DataFrame({
            'Episodes': [12, 24, 13, 25, 12],
            'Popularity': [1000, 2000, 1500, 3000, 1200],
            'Score': [8.5, 7.5, 9.0, 8.0, 7.0],
            'Demographic': [1, 2, 1, 3, 2]
        })
        
        with patch('src.models.get_cached_dataframe', side_effect=[mock_df, mock_numeric]):
            recommender = ContentRecommender()
            recommendations = recommender.get_recommendations(1, top_n=3)
            
            assert isinstance(recommendations, list)
            assert len(recommendations) <= 3
            assert 1 not in recommendations  # L'anime d'entrée ne doit pas être recommandé
    
    @patch('src.models.get_cached_dataframe')
    @patch('src.models.np.load')
    def test_get_recommendations_invalid_id(self, mock_np_load, mock_get_df):
        """Test avec un ID invalide."""
        mock_df = pd.DataFrame({
            'MAL_ID': [1, 2, 3],
            'Title': ['A', 'B', 'C']
        })
        mock_get_df.return_value = mock_df
        
        recommender = ContentRecommender()
        
        with pytest.raises(ValueError):
            recommender.get_recommendations(999)  # ID qui n'existe pas


class TestFPGrowthRecommender:
    """Tests pour le FPGrowthRecommender."""
    
    @patch('src.models.get_cached_dataframe')
    def test_initialization(self, mock_get_df):
        """Test que le recommender s'initialise correctement."""
        recommender = FPGrowthRecommender()
        assert recommender._ratings is None
        assert recommender._rules is None
    
    @patch('src.models.get_cached_dataframe')
    def test_get_recommendations(self, mock_get_df):
        """Test génération de recommandations."""
        # Mock des règles
        mock_rules = pd.DataFrame({
            'antecedents': ["frozenset({1})", "frozenset({2})"],
            'consequents': ["frozenset({3})", "frozenset({4})"],
            'support': [0.5, 0.3]
        })
        
        # Simuler le chargement des données
        def side_effect(table_name, ttl):
            if table_name == "association_rules":
                return mock_rules
            elif table_name == "anime_transactions":
                return pd.DataFrame({'user_id': [1], 'MAL_ID': [1], 'rating': [8]})
        
        mock_get_df.side_effect = side_effect
        
        recommender = FPGrowthRecommender()
        recommendations = recommender.get_recommendations(1)
        
        assert isinstance(recommendations, list)


class TestUtilityFunctions:
    """Tests pour les fonctions utilitaires."""
    
    def test_find_duplicates(self):
        """Test de la fonction find_duplicates."""
        # Test basique
        input_list = [1, 2, 3, 2, 4, 3, 5]
        result = find_duplicates(input_list)
        assert sorted(result) == [2, 3]
        
        # Test sans doublons
        input_list = [1, 2, 3, 4, 5]
        result = find_duplicates(input_list)
        assert result == []
        
        # Test tous doublons
        input_list = [1, 1, 2, 2, 3, 3]
        result = find_duplicates(input_list)
        assert sorted(result) == [1, 2, 3]
    
    def test_combine_results(self):
        """Test de la fonction combine_results."""
        list1 = [1, 2, 3, 4, 5]  # Collaborative
        list2 = [6, 7, 8, 9, 10]  # Content
        
        mock_data = pd.DataFrame({
            'MAL_ID': list(range(1, 11))
        })
        
        result = combine_results(list1, list2, mock_data, k=10)
        
        # Doit retourner 5 de list1 et 5 de list2
        assert len(result) == 10
        assert result[:5] == [1, 2, 3, 4, 5]
        assert result[5:] == [6, 7, 8, 9, 10]
    
    def test_combine_results_with_overlap(self):
        """Test combine_results avec des éléments communs."""
        list1 = [1, 2, 3, 4, 5]
        list2 = [3, 4, 5, 6, 7, 8]  # 3, 4, 5 sont en double
        
        mock_data = pd.DataFrame({
            'MAL_ID': list(range(1, 9))
        })
        
        result = combine_results(list1, list2, mock_data, k=8)
        
        # Pas de doublons
        assert len(result) == len(set(result))
        # Les premiers éléments viennent de list1
        assert result[:5] == [1, 2, 3, 4, 5]


@pytest.fixture
def sample_anime_data():
    """Fixture pour des données d'anime de test."""
    return pd.DataFrame({
        'MAL_ID': [1, 2, 3, 4, 5],
        'Title': ['Naruto', 'One Piece', 'Bleach', 'Death Note', 'Code Geass'],
        'Score': [8.3, 8.7, 7.9, 9.0, 8.7],
        'Release': [2002, 1999, 2004, 2006, 2006]
    })


@pytest.fixture
def mock_recommender():
    """Fixture pour un recommender mocké."""
    with patch('src.models.get_cached_dataframe'):
        return ContentRecommender()
