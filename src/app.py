# app.py

"""
Streamlit web application for the RAG Chatbot.

This script creates a user-friendly, chat-based interface for interacting
with the RAG pipeline served by the FastAPI backend.

Features:
- Conversation memory with session persistence
- Enhanced responses with confidence scores and sources
- Multi-part question handling
- Follow-up question support
"""

import streamlit as st
import requests
import os
import uuid

# SECTION 1: API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ASK_ENDPOINT = f"{API_BASE_URL}/ask/enhanced"  # Use enhanced endpoint
ASK_BASIC_ENDPOINT = f"{API_BASE_URL}/ask"  # Fallback to basic endpoint

# Page Configuration
st.set_page_config(
    page_title="XMUM AI Chatbot",
    page_icon="🎓",
    layout="centered"
)


# SECTION 2: SESSION MANAGEMENT
# Generate or retrieve session ID for conversation memory
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "use_enhanced" not in st.session_state:
    st.session_state.use_enhanced = True


# SECTION 3: UI HEADER
st.title("🎓 XMUM AI Campus Chatbot")
st.markdown("""
Welcome! I'm an AI assistant trained on XMUM documents.
Ask me anything about campus rules, academic policies, fees, or procedures.
""")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.use_enhanced = st.checkbox(
        "Enhanced Mode", 
        value=st.session_state.use_enhanced,
        help="Enable conversation memory, multi-part questions, and confidence scores"
    )
    
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        # Clear server-side session too
        try:
            requests.delete(f"{API_BASE_URL}/session/{st.session_state.session_id}", timeout=5)
        except:
            pass
        st.rerun()
    
    st.markdown("---")
    st.caption(f"Session: {st.session_state.session_id[:8]}...")

st.markdown("---")


# SECTION 4: CHAT HISTORY DISPLAY
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show metadata for assistant messages in enhanced mode
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            if meta.get("sources"):
                with st.expander("📚 Sources"):
                    for source in meta["sources"]:
                        st.caption(f"• {source}")
            if meta.get("confidence"):
                confidence = meta["confidence"]
                color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                st.caption(f"Confidence: :{color}[{confidence:.0%}]")


# SECTION 5: CHAT INPUT AND RESPONSE
if prompt := st.chat_input("Ask a question about XMUM..."):
    # 1. Add user's message to the chat history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Display a spinner while waiting for the API response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_response = ""
            metadata = {}
            
            try:
                if st.session_state.use_enhanced:
                    # Use enhanced endpoint with session ID
                    request_data = {
                        "query": prompt,
                        "session_id": st.session_state.session_id
                    }
                    response = requests.post(ASK_ENDPOINT, json=request_data, timeout=90)
                    
                    if response.status_code == 200:
                        data = response.json()
                        full_response = data.get("response", "Sorry, I couldn't find an answer.")
                        metadata = {
                            "sources": data.get("sources", []),
                            "confidence": data.get("confidence", 1.0),
                            "needs_clarification": data.get("needs_clarification", False),
                            "clarification_prompt": data.get("clarification_prompt"),
                        }
                        
                        # If clarification is needed, add it to the response
                        if metadata.get("needs_clarification") and metadata.get("clarification_prompt"):
                            full_response += f"\n\n💡 **Clarification needed:** {metadata['clarification_prompt']}"
                    else:
                        # Fallback to basic endpoint
                        request_data = {"query": prompt}
                        response = requests.post(ASK_BASIC_ENDPOINT, json=request_data, timeout=60)
                        if response.status_code == 200:
                            full_response = response.json().get("response", "Sorry, I couldn't find an answer.")
                        else:
                            full_response = f"Error: Server responded with status {response.status_code}."
                else:
                    # Basic mode
                    request_data = {"query": prompt}
                    response = requests.post(ASK_BASIC_ENDPOINT, json=request_data, timeout=60)
                    
                    if response.status_code == 200:
                        full_response = response.json().get("response", "Sorry, I couldn't find an answer.")
                    else:
                        error_details = response.text
                        full_response = f"Error: Status {response.status_code}. {error_details}"
            
            except requests.exceptions.Timeout:
                full_response = "⏱️ Request timed out. Please try again with a simpler question."
            except requests.exceptions.RequestException as e:
                full_response = f"❌ Could not connect to the chatbot backend. Please ensure the server is running.\n\nDetails: {e}"
            
            # 3. Display the response
            st.markdown(full_response)
            
            # 4. Show metadata if available
            if metadata.get("sources"):
                with st.expander("📚 Sources"):
                    for source in metadata["sources"]:
                        st.caption(f"• {source}")
            if metadata.get("confidence"):
                confidence = metadata["confidence"]
                color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                st.caption(f"Confidence: :{color}[{confidence:.0%}]")
    
    # 5. Add the response to the chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "metadata": metadata
    })