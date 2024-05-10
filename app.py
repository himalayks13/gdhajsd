import streamlit as st
import requests

# URL of your FastAPI backend
API_URL = "http://localhost:8000"

def create_user(username):
    """Send a POST request to the backend to create a user."""
    response = requests.post(f"{API_URL}/create-user/", json={"username": username})
    return response.json()

def create_session(user_id):
    """Send a POST request to the backend to create a session."""
    response = requests.post(f"{API_URL}/create-session/", json={"user_id": user_id})
    return response.json()

def upload_document(file):
    """Send a POST request to upload a document."""
    files = {'file': file.getvalue()}
    response = requests.post(f"{API_URL}/upload/", files=files)
    return response.json()

def post_query(query, session_id):
    """Send a POST request to get an answer for a query."""
    response = requests.post(f"{API_URL}/post-query/", json={"query": query, "session_id": session_id})
    return response.json()

# Streamlit layout
st.title('Welcome to the Intelligent Query Processing System')

# User creation form
with st.form("user_form"):
    username = st.text_input("Enter your username")
    submit_user = st.form_submit_button("Create User")
    if submit_user:
        user_response = create_user(username)
        if "user_id" in user_response:
            user_id = user_response["user_id"]
            st.success(f"User created successfully with ID: {user_id}")
            # Session creation once the user is created
            session_response = create_session(user_id)
            if "session_id" in session_response:
                session_id = session_response["session_id"]
                st.success(f"Session started with ID: {session_id}")
        else:
            st.error("Failed to create user or start session")

# Document upload
st.header("Upload Documents")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    upload_response = upload_document(uploaded_file)
    if upload_response:
        st.write(upload_response)

# Chat with the chatbot
st.header("Chat with the Chatbot")
query = st.text_input("Ask a question")
if st.button("Send"):
    if query:
        response = post_query(query, session_id)
        st.text("Response from the chatbot:")
        st.write(response["answer"])
    else:
        st.error("Please type a question.")
