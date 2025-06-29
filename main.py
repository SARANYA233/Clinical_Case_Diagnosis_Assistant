import streamlit as st
from src.utils.predictor import predict_disease

st.set_page_config(page_title="Clinical Case Diagnosis", layout="centered")

st.title("🧠 Clinical Case Diagnosis")
st.markdown("Enter your symptoms and get possible disease predictions using AI and LLMs.")

# 📝 Input field for user
user_input = st.text_area("Describe your symptoms in your own words:")

# 🔘 Prediction button
if st.button("Predict Disease"):
    with st.spinner("Analyzing symptoms..."):
        st.write("Predicting in progress... ⏳")
        Disease, Symptoms = predict_disease(user_input)

    if not Disease:
        st.error("No symptoms were detected. Please try rephrasing.")
    else:
        st.success("Prediction complete! ✅")
        st.markdown("**Matched Symptoms**:")
        st.write(", ".join(Symptoms))

        st.markdown("### 🔍 Predicted Diseases")
        for disease, prob in Disease:
            st.write(f"**{disease}** — {prob}%")

