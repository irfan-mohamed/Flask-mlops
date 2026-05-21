from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

with open("models/iris_pipeline.pkl", 'rb') as f :
    model = joblib.load(f)

@app.route('/')
def home():
    return jsonify({
        'message' : 'Kmeans API is Running'
    })

@app.route('/predict', methods = ["POST"])
def predict():
    try :
        data = request.get_json()
        if 'features' not in data :
            return jsonify({
                'error' : 'features are missing'
            }), 400

        features = data['features']

        if not isinstance(features, list):
            return jsonify({
                'error' : 'features must be a list'
            }), 400
        
        if len(features) != 4 :
            return jsonify({
                'error' : 'Exactly 4 features are needed'
            }), 400
        
        validated_features = []
    
        for value in features:
            if value is None:
                return jsonify({
                    'error' : 'The Value Should Not Be Null'
                }), 400
            elif not isinstance(value, (int, float)):
                return jsonify({
                    'error' : 'Feature must be int or float'
                }), 400 
            validated_features.append(float(value))



        features_n = np.array(validated_features).reshape(1, -1)

        prediction = model.predict(features_n)

        return jsonify({
            'prediction' : int(prediction[0])
        }), 200

    except Exception as e:
        return jsonify({
            'Internal servor error' : str(e)
        }), 500


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000, debug = True)