import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Désactiver rate limiting pour les tests
os.environ['TESTING'] = '1'

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test que la page d'accueil renvoie un statut 200"""
    response = client.get('/')
    assert response.status_code == 200

def test_search_anime(client):
    """Test que la recherche d’anime fonctionne"""
    response = client.get('/search_animes', query_string={'query': 'Naruto'})
    assert response.status_code == 200
    assert isinstance(response.json, list)
