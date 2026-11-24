import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("verifact_model.pkl")
tfidf = joblib.load("verifact_tfidf.pkl")

# Page configuration
st.set_page_config(
    page_title="VeriFact",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 VeriFact")
st.subheader("Real or Fake News Detection")

st.write(
    "Enter a news headline below to check whether it is real or fake."
)

# User input
news = st.text_area(
    "Enter News Headline:",
    placeholder="Type your news headline here..."
)

# Prediction button
if st.button("Check News"):

    if news.strip():

        # Convert text into TF-IDF
        news_tfidf = tfidf.transform([news])

        # Predict
        prediction = model.predict(news_tfidf)[0]

        if int(prediction) == 0:

            st.error(" Prediction: Fake News")

        else:

            st.success(" Prediction: Real News")

    else:

        st.warning("Please enter a news headline!")