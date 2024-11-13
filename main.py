import time
import random
import streamlit as st
from transformers import pipeline


# Streamlit app
def main():
    # Page title
    st.title("Emotion Analyzer")
    # User input
    input_text = st.text_input("Enter your text here:")

    if st.button("Submit"):
        if input_text:
            with st.spinner("Loading models and processing..."):
                emotion_recog_model = pipeline(
                    "text-classification", model="PriyHF/emotion_recog"
                )
                brand_product_category_recog_model = pipeline(
                    "text-classification", model="PriyHF/brand_product_recog"
                )

                emotion_pred = emotion_recog_model(input_text)
                st.subheader("Predicted Emotion: ")
                st.write(emotion_pred)

                if emotion_pred:
                    brand_product_category_pred = brand_product_category_recog_model(
                        input_text
                    )
                    st.subheader("Category: ")
                    st.write(brand_product_category_pred)
                else:
                    st.warning("Input classified as Type B")
        else:
            st.error("Please enter some text!")


if __name__ == "__main__":
    main()
