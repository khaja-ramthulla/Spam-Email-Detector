from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    email_text = request.form.get("email", "").strip()

    if not email_text:
        return jsonify({
            "error": "Please enter some text."
        }), 400

    transformed = vectorizer.transform([email_text])

    prediction = model.predict(transformed)[0]

    probabilities = model.predict_proba(transformed)[0]

    confidence = round(float(max(probabilities) * 100), 2)

    suspicious_words = [
        "free", "winner", "won", "prize", "claim",
        "urgent", "offer", "cash", "money", "click"
    ]

    found_words = []

    text_lower = email_text.lower()

    for word in suspicious_words:
        if word in text_lower:
            found_words.append(word)

    return jsonify({
        "prediction": prediction.upper(),
        "confidence": confidence,
        "keywords": found_words
    })


if __name__ == "__main__":
    app.run(debug=True)