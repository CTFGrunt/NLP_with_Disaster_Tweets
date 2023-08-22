from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Chargement du modèle Naive Bayes
model_MNB = joblib.load("naive_bayes_multinomial_model.pkl")  # Remplacez par le chemin vers votre modèle Naive Bayes
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")  # Remplacez par le chemin vers votre vecteuriseur TF-IDF

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    # Prétraitement du texte (vectorisation TF-IDF, etc.)
    text_tfidf = tfidf_vectorizer.transform([text])

    # Faire la prédiction en utilisant le modèle
    prediction = model_MNB.predict(text_tfidf)[0]

    # Convertir la prédiction en int
    prediction = int(prediction)

    # Renvoyer la réponse JSON avec la prédiction
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
