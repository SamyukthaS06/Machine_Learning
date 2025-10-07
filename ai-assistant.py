import streamlit as st
import google.generativeai as genai
from PIL import Image

st.set_page_config(page_title="Gemini Chat Assistant", layout="centered")

# Load and display the robot image
st.image("robo.jpg", width=800)

# Title and subtitle
st.markdown("""
    <h1 style='text-align: center; font-family: Segoe UI, sans-serif;'>🤖 Gemini Chat Assistant</h1>
    <p style='text-align: center; color: #30d158;'>Ask your health or disease-related questions below.</p>
""", unsafe_allow_html=True)

# Styling (optional highlight colors)
st.markdown("""
    <style>
    .chat-response {
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-family: 'Segoe UI', sans-serif;
        color: white;
        background-color: #262730;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Chat input
user_question = st.text_input("💬 Type your question here:")

# Load Gemini model once
if "chat_model" not in st.session_state:
    st.session_state.chat_model = genai.GenerativeModel("gemini-1.5-flash").start_chat()

# Process response
if user_question:
    with st.spinner("🤖 Gemini is thinking..."):
        try:
            st.session_state.chat_model.send_message(user_question)
            response = st.session_state.chat_model.last.text
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"

    st.markdown(f"<div class='chat-response'><strong>🧑‍⚕️ You:</strong> {user_question}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-response'><strong>🤖 Gemini:</strong> {response}</div>", unsafe_allow_html=True)

# Back button
if st.button("⬅ Back to Home"):
    st.switch_page("final.py")
