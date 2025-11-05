from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy
import os


app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load('logistic_regression_model.joblib')  # Adjust filename
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # If you have a vectorizer 

@app.route('/', methods=['GET'])
def init():
    return ('Hello World')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        email_text = data.get('email', '')
        if not email_text:
            return jsonify({'error': 'No email text provided'})
        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]
        probability = model.predict_proba(email_vector)[0]
        result ={
            'is_spam': bool(prediction),
            'confidence': float(max(probability)),
            'ham_probablity': float(probability[0]),
             'spam_probability': float(probability[1])

        }      
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="localhost", port=8000, debug=True)