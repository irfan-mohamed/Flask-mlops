from app import app

client = app.test_client()

def test_missing_features():
    response = client.post('/predict', json = {})

    assert response.status_code == 400

    assert b'features are missing' in response.data