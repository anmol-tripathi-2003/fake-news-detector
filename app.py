import streamlit as st
import pickle
import os

st.title("📰 Fake News Detector")

# Debug section
st.write("Current working directory:", os.getcwd())
st.write("Files:", os.listdir())

# Try to load files
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    st.success("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# UI
st.subheader("Enter a news article below to predict whether it's real or fake.")
text = st.text_area("Paste the news article text here", height=300)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]
        if prediction == 1:
            st.success("🟢 Prediction: This appears to be **True News**.")
        else:
            st.error("🔴 Prediction: This appears to be **Fake News**.")
