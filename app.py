import os
from dotenv import load_dotenv
import streamlit as st
import webchat
import utils

# URL of the hosted LLMs is hardcoded because at this time all LLMs share the same endpoint
url = "https://us-south.ml.cloud.ibm.com"

def main():
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("api_key")
    project_id = os.getenv("project_id")

    if not api_key or not project_id:
        st.error("Missing API Key or Project ID in the .env file.")
        return

    st.set_page_config(layout="wide", page_title="RAG Web Demo", page_icon="")
    utils.load_css("styles.css")
    
    # Streamlit app title with style
    st.markdown("""
        <div class="menu-bar">
            <h1>IBM watsonx.ai - webchat</h1>
        </div>
        <div style="margin-top: 20px;"><p>Insert the website you want to chat with and ask your question.</p></div>
    """, unsafe_allow_html=True)
    
    # Sidebar for information
    st.sidebar.header("Information")
    st.sidebar.markdown("Credentials are securely loaded from your `.env` file. Ensure the file is properly configured.", unsafe_allow_html=True)
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    
    # Main input area
    user_url = st.text_input('Provide a URL')
    # UI component to enter the question
    question = st.text_area('Question', height=100)
    button_clicked = st.button("Answer the question")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Response")
    
    collection_name = "base"
    client = utils.chromadb_client()
    
    if button_clicked and user_url:
        # Invoke the LLM when the button is clicked
        response = webchat.answer_questions_from_web(api_key, project_id, user_url, question, collection_name, client)
        st.write(response)
    else:
        if not user_url and button_clicked:
            st.warning("Please provide a URL to proceed.")
  
    # Cleaning Vector Database
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.header("Memory")
    clean_button_clicked = st.sidebar.button("Clean Memory")
    if clean_button_clicked:
        if collection_name:
            utils.clear_collection(collection_name, client)
            st.sidebar.success("Memory cleared successfully!")
            print("Memory cleared successfully!")
        else:
            st.sidebar.error("Collection name is not defined or empty.")

if __name__ == "__main__":
    main()
