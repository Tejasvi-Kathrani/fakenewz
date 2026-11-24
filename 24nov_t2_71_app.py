from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("verifact_model.pkl")
tfidf = joblib.load("verifact_tfidf.pkl")


@app.route('/')
def home():
    return "VeriFact Fake News Detection API is Running!"


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "Please provide JSON data"}), 400

    news = data.get('news') or data.get('title')

    if not news or not isinstance(news, str) or not news.strip():
        return jsonify({"error": "Please enter a valid news headline"}), 400

    # Convert text into TF-IDF
    news_tfidf = tfidf.transform([news])

    # Predict
    prediction = model.predict(news_tfidf)[0]

    if int(prediction) == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return jsonify({
        "news": news,
        "prediction": result
    })


if __name__ == '__main__':
    app.run(debug=True)