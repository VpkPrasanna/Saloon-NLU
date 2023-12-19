import streamlit as st
from transformers import pipeline

# Load the text classification pipeline from Hugging Face
classifier = pipeline("text-classification", model="test1/")


def main():
    st.title("Text Classification with Hugging Face Transformers")

    # Get user input
    user_input = st.text_area("Enter text for classification:")

    # Perform classification when user clicks the button
    if st.button("Classify"):
        if user_input:
            # Make prediction using the Hugging Face pipeline
            result = classifier(user_input)

            # Display the result
            st.json(result)
            # st.success(
            #     f"Predicted Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}")
        else:
            st.warning("Please enter some text for classification.")


if __name__ == "__main__":
    main()
