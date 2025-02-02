import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_recommendations(client):
    """Test la route des recommandations"""
    payload = {"titles": ["Naruto"]}
    response = client.post('/recommendations', json=payload)
    assert response.status_code == 200
    assert isinstance(response.json, list)
