# app.py

"""
Streamlit web application for the RAG Chatbot.

This script creates a user-friendly, chat-based interface for interacting
with the RAG pipeline served by the FastAPI backend
"""

import streamlit as st
import requests
import os

# SECTION 1
# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ASK_ENDPOINT = f"{API_BASE_URL}/ask"

# Page Configuration
st.set_page_config(
    page_title="XMUM AI Chatbot",
    page_icon="",
    layout="centered"
)


# SECTION 2: UI
st.title("XMUM AI Campus Chatbot")
st.markdown("""
Welcome! I'm an AI assistant trained on the XMUM student handbook.
Ask me anything about campus rules, academic policies, or any procedures.
""")
st.markdown("---")

# Streamlit reruns the script on each interaction, then we use session_state
# to persist data like the chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about XMUM..."):
    # 1. Add user's message to the chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Display a spinner while waiting for the API response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_response = ""
            try:
                # 3. Call the FastAPI
                request_data = {"query": prompt}
                response = requests.post(ASK_ENDPOINT, json=request_data, timeout=60) # Set a timeout
                
                # 4. Handle response
                if response.status_code == 200:
                    full_response = response.json().get("response", "Sorry, I couldn't find an answer.")
                else:
                    # Error message
                    error_details = response.text
                    full_response = f"Error: The server responded with status code {response.status_code}.\nDetails: {error_details}"
            
            except requests.exceptions.RequestException as e:
                # Handle network-related errors
                full_response = f"Error: Could not connect to the chatbot backend. Please ensure the server is running. Details: {e}"
            
            # 5. Display the final response
            st.markdown(full_response)
    
    # 6. Add the response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})