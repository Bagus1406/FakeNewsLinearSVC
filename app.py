import streamlit as st
import pickle

# Load model and label encoder
@st.cache_resource
def load_model():
    with open("svc_fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# Prediction function
def predict(text):
    pred = model.predict([text])[0]
    label = le.inverse_transform([pred])[0]
    return label, 100.0

# Streamlit UI
def main():
    st.markdown(
        """<div style="background-color:#2c3e50;padding:10px;border-radius:10px">
            <h1 style="color:white;text-align:center">Fake News Detector</h1> 
            <h4 style="color:white;text-align:center">Built with Scikit-Learn & Streamlit</h4> 
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """### Welcome!
This app uses a **TF-IDF + LinearSVC** model to classify news content as **FAKE** or **REAL**.  
#### Model Info  
Trained locally using cleaned and preprocessed English-language fake news dataset.
""",
        unsafe_allow_html=True,
    )

    menu = ["Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Prediction":
        st.subheader("Fake News Prediction")
        user_input = st.text_area("Enter a news headline or article (English only):")
        if st.button("Detect"):
            if user_input.strip() == "":
                st.warning("Text cannot be empty.")
            else:
                label, confidence = predict(user_input)
                if label.upper() == "REAL":
                    st.success(f"Prediction: **{label}** ({confidence}% confident)")
                else:
                    st.error(f"Prediction: **{label}** ({confidence}% confident)")
    else:
        st.subheader("Home")

    # Credit
    st.markdown("""---""")
    st.markdown(
        """
        <div style="text-align: center; font-size: 14px;">
            <p><strong>Created by Team Sigma Male</strong></p>
            <ul style="list-style-type: none; padding: 0;">
                <li>1. Hafidz Akbar Faridzi R.</li>
                <li>2. Muhammad Bagus Kurniawan</li>
                <li>3. Nurul Alpi Najam</li>
                <li>4. Ryan Rasyid Azizi</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
