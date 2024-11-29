import streamlit as st
import requests

# Hardcoded credentials
API_KEY = "Q2uZJDscw55ZJ6IGrpOyHw4c7RpkJyY-z6GKIH5Qj--s"
PROJECT_ID = "8e75587d-5ef2-4d94-a5d1-9493d6145ac3"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

# Initialize session state for credentials and URL
if "watsonx_project_id" not in st.session_state:
    st.session_state["watsonx_project_id"] = PROJECT_ID
if "api_key" not in st.session_state:
    st.session_state["api_key"] = API_KEY
if "watsonx_url" not in st.session_state:
    st.session_state["watsonx_url"] = WATSONX_URL

# Watsonx LLM API Call
def call_watsonx_llm(prompt):
    url = f"{st.session_state['watsonx_url']}/v1/projects/{st.session_state['watsonx_project_id']}/generate"

    headers = {
        "Authorization": f"Bearer {st.session_state['api_key']}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": prompt,
        "model": "watsonx_model_name",  # Replace with your Watsonx LLM model name
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 100
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["generated_text"]
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Function to set up the chat interface
def setup_streamlit_chat():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat messages from history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user message
    if prompt := st.chat_input("Type your message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from Watsonx LLM API
        response = call_watsonx_llm(prompt)

        # Add response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display response in chat
        with st.chat_message("assistant"):
            st.markdown(response)

# Main function to render the chat interface
def main():
    st.title("Watsonx LLM Chat Interface")
    st.sidebar.title("Settings")
    st.sidebar.write(f"Project ID: {st.session_state['watsonx_project_id']}")
    st.sidebar.write(f"API Key: {st.session_state['api_key']}")
    st.sidebar.write(f"Watsonx URL: {st.session_state['watsonx_url']}")

    # Set up the chat interface
    setup_streamlit_chat()

if __name__ == "__main__":
    main()
