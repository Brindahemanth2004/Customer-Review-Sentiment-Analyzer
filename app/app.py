import streamlit as st
import joblib
import sys
import os

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Fix path issue
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from utils.preprocessing import clean_text

# ---------------- LOAD MODEL ----------------

model = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# ---------------- VADER ----------------

analyzer = SentimentIntensityAnalyzer()

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon=" "
)

st.title("Customer Review Sentiment Analyzer")

st.write(
    "Analyze reviews using Machine Learning + NLP"
)

# ---------------- INPUT ----------------

user_input = st.text_area(
    "Enter your review:"
)

# ---------------- PREDICTION ----------------

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter a review")

    else:

        # Preprocess text
        cleaned = clean_text(user_input)

        vector = vectorizer.transform([cleaned])

        ml_prediction = model.predict(vector)[0]

        # VADER analysis
        scores = analyzer.polarity_scores(user_input)

        compound = scores['compound']

        text = user_input.lower()

        # ---------------- NEUTRAL DETECTION ----------------

        neutral_phrases = [
            "okay",
            "average",
            "fine",
            "normal",
            "decent",
            "not bad",
            "not good",
            "could be better",
            "nothing special",
            "manageable",
            "acceptable"
        ]

        # Neutral rules
        if any(phrase in text for phrase in neutral_phrases):

            final_prediction = "neutral"

        elif -0.2 <= compound <= 0.2:

            final_prediction = "neutral"

        elif compound > 0.2:

            final_prediction = "positive"

        elif compound < -0.2:

            final_prediction = "negative"

        else:

            final_prediction = ml_prediction

        # ---------------- DISPLAY ----------------

        st.subheader("Analysis Result")

        if final_prediction == "positive":

            st.success("Positive Review")

        elif final_prediction == "negative":

            st.error("Negative Review")

        else:

            st.info("Neutral Review")