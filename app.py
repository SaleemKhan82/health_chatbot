import streamlit as st
from transformers import pipeline

# Load the model
qa_model = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# Define chatbot function
def ask_chatbot(question):
    context = """
    I am not a medical expert, but symptoms like sore throat, fever, cough, or fatigue could indicate common infections. 
    It is best to see a doctor for a thorough diagnosis.
    """
    result = qa_model(question=question, context=context)
    return result['answer']

# Streamlit app interface
st.title('Health Symptom Chatbot')

# Input from the user
user_question = st.text_input('Enter your symptoms or question:')

if user_question:
    # Get response from the chatbot
    answer = ask_chatbot(user_question)
    st.write(f"Chatbot's response: {answer}")
    st.write("Please consult a doctor for further advice.")
