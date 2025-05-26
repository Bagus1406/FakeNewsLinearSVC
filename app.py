import streamlit as st
import pickle
import numpy as np

# Load model and label encoder
@st.cache_resource
def load_model():
    with open("svc_fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, le = load_model()

# Prediction with pseudo-probability using decision function
def predict_with_prob(text):
    decision = model.decision_function([text])[0]
    prob_fake = 1 / (1 + np.exp(decision))         # sigmoid untuk kelas FAKE
    prob_real = 1 - prob_fake
    return [prob_fake, prob_real]

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
                probs = predict_with_prob(user_input)
                labels_map = {0: "FAKE", 1: "REAL"}
                pred_label = labels_map[int(np.argmax(probs))]
                confidence = round(max(probs) * 100, 2)

                if pred_label == "REAL":
                    st.success(f"Prediction: **{pred_label}** ({confidence}% confident)")
                else:
                    st.error(f"Prediction: **{pred_label}** ({confidence}% confident)")

                st.markdown("#### Confidence per class")
                st.bar_chart({
                    "Confidence": {
                        "FAKE": probs[0],
                        "REAL": probs[1]
                    }
                })

    # Credit (let it stay outside the if)
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
