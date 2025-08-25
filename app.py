from flask import Flask, render_template, request
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load logistic regression model & vectorizer
log_reg_model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# VADER analyzer
analyzer = SentimentIntensityAnalyzer()

@app.route("/", methods=["GET", "POST"])
def index():
    logistic_pred = None
    vader_pred = None
    user_input = ""
    logistic_probas = None
    vader_scores = None

    if request.method == "POST":
        user_input = request.form["text"]

        # Logistic Regression prediction
        X = vectorizer.transform([user_input])
        log_reg_output = log_reg_model.predict(X)[0]
        probas = log_reg_model.predict_proba(X)[0]
        classes = log_reg_model.classes_

        logistic_probas = dict(zip(classes, probas))
        logistic_pred = log_reg_output

        # VADER prediction
        vader_scores = analyzer.polarity_scores(user_input)
        compound = vader_scores["compound"]

        if compound >= 0.05:
            vader_pred = "positive"
        elif compound <= -0.05:
            vader_pred = "negative"
        else:
            vader_pred = "neutral"

    return render_template(
        "index.html",
        logistic_pred=logistic_pred,
        vader_pred=vader_pred,
        logistic_probas=logistic_probas,
        vader_scores=vader_scores,
        user_input=user_input,
    )

if __name__ == "__main__":
    app.run(debug=True)
