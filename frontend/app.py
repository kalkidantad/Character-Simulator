import streamlit as st
import requests
import re 



# This script provides a user-friendly interface for interacting with 
# AI-generated characters. Users can upload books, select characters, 
# and engage in dynamic conversations.

#  Features:
# - Upload books for character extraction
# - Select a character from extracted list
# - Converse with AI-powered characters
# - Adjust personality traits using sliders
# - Retain conversation history using Pinecone





# Set page config
st.set_page_config(page_title="AI Character Chatbot", layout="wide")

# Main Chat Interface
st.title("Chat with a Character")

# Sidebar: Upload Book and Display Characters
st.sidebar.title("Upload a Book")
uploaded_file = st.file_uploader("Upload a book (PDF or EPUB)", type=["pdf", "epub"])
user_text = st.text_area("Or enter text here")

# Initialize session state for chat history and selected character
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_character" not in st.session_state:
    st.session_state.selected_character = None

# Process uploaded file and extract characters
if uploaded_file or user_text:
    if st.button("Extract Characters"):
        if uploaded_file:
            BACKEND_URL = "http://127.0.0.1:5001"  

            # Send file to backend for processing
            files = {"file": uploaded_file}
            response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=500)
        else:
            # Send text to backend for processing
            response = requests.post("http://localhost:5001/upload", json={"text": user_text})
        
        if response.status_code == 200:
            characters = response.json().get("characters", [])
            st.session_state.characters = characters
        else:
            st.error("Failed to extract any characters.")



# Display extracted characters as buttons
if "characters" in st.session_state:
    st.write("### Select a Character to Chat With")
    for character in st.session_state.characters:
        if st.button(character):
            st.session_state.selected_character = character




# Display selected character
if st.session_state.selected_character:
    st.write(f"**Chatting as {st.session_state.selected_character}**")

    # Display chat history FIRST (before input)
    for speaker, message in st.session_state.chat_history:
        st.write(f"**{speaker}:** {message}")

    # User input at BOTTOM (avoids refresh issues)
    user_input = st.text_input("Your Message:", key="chat_input")
    
    if user_input and not st.session_state.get("waiting_for_response"):
        st.session_state.waiting_for_response = True
        
        # Add user message immediately
        st.session_state.chat_history.append(("You", user_input))
        
        with st.spinner(f"{st.session_state.selected_character} is typing..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:5001/chat",
                    json={
                        "message": user_input,
                        "character": st.session_state.selected_character
                    },
                    timeout=100
                )
                
                if response.status_code == 200:
                    raw_json = response.json()
                    st.write(f"Debug: {raw_json}")  # TEMPORARY DEBUG LINE

                    # Extract response from JSON
                    reply = raw_json.get("response", "").strip()

                    # Remove unwanted system tags
                    reply = re.sub(r"<<SYS>>|<</SYS>>", "", reply).strip()

                    if reply:
                        st.session_state.chat_history.append(
                            (st.session_state.selected_character, reply)
                        )
                    else:
                        st.warning(f"Empty response from backend! Full response: {raw_json}")
                else:
                    st.error(f"Backend error: {response.text}")

                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection failed: {str(e)}")
            finally:
                st.session_state.waiting_for_response = False
                # NO st.rerun() needed - Streamlit auto-updates on state change

# Sidebar: Psychological Parameters and Emotional States
st.sidebar.subheader("Psychological Parameters")
params = {
    "Valence Level": 50,
    "Arousal Level": 50,
    "Selection Threshold": 50,
    "Resolution Level": 50,
    "Goal-Directedness": 50,
    "Securing Rate": 50
}

# Update sliders based on backend response
if st.session_state.selected_character:
    response = requests.get("http://127.0.0.1:5001/get_parameters", params={"character": st.session_state.selected_character})
    if response.status_code == 200:
        params = response.json().get("psychological_params", params)

for param, value in params.items():
    params[param] = st.sidebar.slider(param, min_value=0, max_value=100, value=value)

st.sidebar.subheader("Emotional States")
emotion_states = {
    "Anger": 0,
    "Sadness": 0,
    "Joy": 0,
    "Pride": 0,
    "Bliss": 0
}


# Update sliders based on backend response
if st.session_state.selected_character:
    response = requests.get("http://127.0.0.1:5001/get_emotions", params={"character": st.session_state.selected_character})
    if response.status_code == 200:
        emotion_states = response.json().get("emotional_states", emotion_states)


for emotion, value in emotion_states.items():
    emotion_states[emotion] = st.sidebar.slider(emotion, min_value=0, max_value=100, value=value)

# Send updated parameters and emotions to backend
if st.sidebar.button("Update Parameters"):
    response = requests.post(
        "http://127.0.0.1:5001/update_parameters",
        json={"character": st.session_state.selected_character, "params": params, "emotions": emotion_states}
    )
    if response.status_code == 200:
        st.sidebar.success("Parameters updated successfully!")
    else:
        st.sidebar.error("Failed to update parameters.")


