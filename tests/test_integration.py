from app import app

client = app.test_client()

def test_prediction_endpoint():

    response = client.post(
        '/predict',
        json={
            "features": [5.1, 3.5, 1.4, 0.2]
        }
    )


    data = response.get_json()

    assert response.status_code == 200

    assert "prediction" in data

