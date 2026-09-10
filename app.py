import streamlit as st
import joblib

st.set_page_config(
    page_title = "Sentiment-Analyser",
    layout = "centered"
)

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("Product Review Sentiment Analyzer")
st.caption("Enter a product review and see whether the model predicts it as positive or negative.")

text = st.text_area("Enter the text to review", placeholder="e.g. This product exceeded my expectations...")

if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter the text to review")
    else:
        text_vec = vectorizer.transform([text])
        y_predict = model.predict(text_vec)[0]
        proba = model.predict_proba(text_vec)
        print("proba:", proba)

        if y_predict == 1:
            st.write("Review is positive")
        else:
            st.write("Review is negative")  
        

