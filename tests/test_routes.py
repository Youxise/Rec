import pytest
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
