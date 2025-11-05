from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allows your frontend

# Load model & vectorizer at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic_regression_model.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.route('/', methods=['GET'])
def init():
    return 'ML API is running!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        email_text = data.get('email', '').strip()
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400

        email_vector = vectorizer.transform([email_text])
        prediction = model.predict(email_vector)[0]
        prob = model.predict_proba(email_vector)[0]

        return jsonify({
            'is_spam': bool(prediction),
            'confidence': float(max(prob)),
            'ham_probability': float(prob[0]),
            'spam_probability': float(prob[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# RENDER REQUIRED: Use $PORT and 0.0.0.0
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)