"""
Tests pour les routes de l'API.

Ces tests vérifient le bon fonctionnement de toutes les routes
de l'application Flask.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
import pandas as pd

# Add project root to path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def client():
    """Fixture pour le client de test Flask."""
    from app import app
    app.config['TESTING'] = True
    app.config['DEBUG'] = False
    
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_animes():
    """Fixture avec des données d'anime de test."""
    return pd.DataFrame({
        'MAL_ID': [1, 2, 3, 4, 5],
        'Title': ['Naruto', 'One Piece', 'Bleach', 'Death Note', 'Code Geass'],
        'Release': [2002, 1999, 2004, 2006, 2006],
        'Score': [8.3, 8.7, 7.9, 9.0, 8.7],
        'Popularity': [50000, 80000, 40000, 90000, 70000],
        'Episodes': [220, 1000, 366, 37, 50]
    })


class TestHealthRoute:
    """Tests pour la route /health."""
    
    def test_health_check_success(self, client):
        """Test que le health check retourne 200 quand tout va bien."""
        with patch('app.check_connection', return_value=True):
            response = client.get('/health')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'healthy'
            assert data['database'] == 'connected'
    
    def test_health_check_db_down(self, client):
        """Test que le health check retourne 503 si la DB est down."""
        with patch('app.check_connection', return_value=False):
            response = client.get('/health')
            assert response.status_code == 503
            data = json.loads(response.data)
            assert data['database'] == 'disconnected'


class TestSearchAnimesRoute:
    """Tests pour la route /search_animes."""
    
    def test_search_animes_success(self, client, sample_animes):
        """Test recherche d'animes avec succès."""
        with patch('app.get_anime_metadata', return_value=sample_animes):
            response = client.get('/search_animes?query=naruto')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data) >= 1
            assert data[0]['Title'] == 'Naruto'
    
    def test_search_animes_empty_query(self, client):
        """Test avec une requête vide."""
        response = client.get('/search_animes?query=')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == []
    
    def test_search_animes_short_query(self, client):
        """Test avec une requête trop courte."""
        response = client.get('/search_animes?query=a')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_search_animes_no_results(self, client, sample_animes):
        """Test quand aucun résultat n'est trouvé."""
        with patch('app.get_anime_metadata', return_value=sample_animes):
            response = client.get('/search_animes?query=xxxnonexistentxxx')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data) == 0


class TestRecommendationsRoute:
    """Tests pour la route /recommendations."""
    
    def test_recommendations_success(self, client, sample_animes):
        """Test génération de recommandations avec succès."""
        payload = {"titles": ["Naruto"]}
        
        with patch('app.get_anime_metadata', return_value=sample_animes), \
             patch('app.recommender_content.get_recommendations', return_value=[2, 3]), \
             patch('app.recommender_collaborative.get_recommendations', return_value=[4, 5]), \
             patch('app.combine_results', return_value=[2, 3, 4, 5]):
            
            response = client.post(
                '/recommendations',
                data=json.dumps(payload),
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, list)
    
    def test_recommendations_invalid_json(self, client):
        """Test avec JSON invalide."""
        response = client.post(
            '/recommendations',
            data="invalid json",
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_recommendations_missing_titles(self, client):
        """Test sans le champ 'titles'."""
        payload = {"wrong_field": ["Naruto"]}
        response = client.post(
            '/recommendations',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_recommendations_empty_titles(self, client):
        """Test avec une liste de titres vide."""
        payload = {"titles": []}
        response = client.post(
            '/recommendations',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_recommendations_too_many_titles(self, client):
        """Test avec trop de titres."""
        payload = {"titles": ["Anime" + str(i) for i in range(10)]}
        response = client.post(
            '/recommendations',
            data=json.dumps(payload),
            content_type='application/json'
        )
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Maximum' in data['error']
    
    def test_recommendations_anime_not_found(self, client, sample_animes):
        """Test avec un anime qui n'existe pas."""
        payload = {"titles": ["NonExistentAnime"]}
        
        with patch('app.get_anime_metadata', return_value=sample_animes):
            response = client.post(
                '/recommendations',
                data=json.dumps(payload),
                content_type='application/json'
            )
            assert response.status_code == 404
            data = json.loads(response.data)
            assert 'non trouvés' in data['error']


class TestDiscoverAnimesRoute:
    """Tests pour la route /discover_animes."""
    
    def test_discover_default_params(self, client, sample_animes):
        """Test découverte avec paramètres par défaut."""
        with patch('app.get_anime_metadata', return_value=sample_animes):
            response = client.get('/discover_animes')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert isinstance(data, list)
            assert len(data) <= 10
    
    def test_discover_sort_by_score(self, client, sample_animes):
        """Test tri par score."""
        with patch('app.get_anime_metadata', return_value=sample_animes):
            response = client.get('/discover_animes?sort_by=Score&order=desc')
            assert response.status_code == 200
            data = json.loads(response.data)
            # Vérifier que c'est trié par score décroissant
            if len(data) > 1:
                assert data[0]['Score'] >= data[1]['Score']
    
    def test_discover_filter_by_year(self, client, sample_animes):
        """Test filtrage par année."""
        with patch('app.get_anime_metadata', return_value=sample_animes):
            response = client.get('/discover_animes?year=2006')
            assert response.status_code == 200
            data = json.loads(response.data)
            # Tous les animes retournés doivent être de 2006
            for anime in data:
                assert anime['Release'] == 2006
    
    def test_discover_invalid_sort_by(self, client):
        """Test avec un champ de tri invalide."""
        response = client.get('/discover_animes?sort_by=InvalidField')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'invalide' in data['error']
    
    def test_discover_invalid_order(self, client):
        """Test avec un ordre invalide."""
        response = client.get('/discover_animes?order=invalid')
        assert response.status_code == 400
    
    def test_discover_with_limit(self, client, sample_animes):
        """Test avec limite personnalisée."""
        with patch('app.get_anime_metadata', return_value=sample_animes):
            response = client.get('/discover_animes?limit=3')
            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data) <= 3


class TestErrorHandlers:
    """Tests pour les gestionnaires d'erreurs."""
    
    def test_404_error(self, client):
        """Test que la route inexistante retourne 404."""
        response = client.get('/nonexistent_route')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_rate_limiting(self, client):
        """Test que le rate limiting fonctionne."""
        # Faire beaucoup de requêtes rapides
        for _ in range(35):  # Limite : 30 per minute pour search
            client.get('/search_animes?query=test')
        
        # La suivante devrait être rate-limitée
        response = client.get('/search_animes?query=test')
        assert response.status_code == 429


class TestIndexRoute:
    """Tests pour la route d'accueil."""
    
    def test_index_page(self, client):
        """Test que la page d'accueil se charge."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'<!DOCTYPE html>' in response.data or b'<html' in response.data
