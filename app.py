import os
from urllib.parse import urlparse
from dotenv import load_dotenv
import streamlit as st
from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
import requests
from bs4 import BeautifulSoup
import spacy
import chromadb

# Initialize global variables
default_url = "https://us-south.ml.cloud.ibm.com"
current_dir = os.getcwd()
cache_dir = os.path.join(current_dir, ".cache")

# Hugging Face Model Cache
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
os.environ["HF_HOME"] = cache_dir
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(model_name, cache_folder=cache_dir)

# Streamlit Config
st.set_page_config(layout="wide", page_title="RAG Web Demo", page_icon="")
css_content = """
.reportview-container, .main {
    background: #ffffff;
    color: #000000;
}
.sidebar .sidebar-content {
    background: #f0f2f6;
    color: #000000;
}
.stButton>button {
    background-color: #0D62FE;
    color: white;
}
.stTextInput>div>div>input {
    color: #000000;
    background-color: #ffffff;
}
.stTextArea>div>textarea {
    color: #000000;
    background-color: #ffffff;
}
"""
st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)


class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = sentence_model

    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()


def chromadb_client():
    settings = chromadb.Settings(persist_directory=cache_dir)
    client = chromadb.Client(settings)
    return client


def clear_collection(collection_name, client):
    try:
        collection = client.get_collection(collection_name)
        if collection:
            collection.delete()
            st.sidebar.success("Memory cleared successfully!")
    except ValueError:
        pass  # Collection does not exist


def extract_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            p_contents = [p.get_text() for p in soup.find_all('p')]
            raw_web_text = " ".join(p_contents)
            return raw_web_text.replace("\xa0", " ")
        else:
            st.error(f"Failed to retrieve the page. Status code: {response.status_code}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def split_text_into_sentences(text):
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def create_embedding(url, collection_name, client):
    cleaned_text = extract_text(url)
    cleaned_sentences = split_text_into_sentences(cleaned_text)
    collection = client.get_or_create_collection(collection_name)
    collection.upsert(
        documents=cleaned_sentences,
        metadatas=[{"source": str(i)} for i in range(len(cleaned_sentences))],
        ids=[str(i) for i in range(len(cleaned_sentences))]
    )
    return collection


def create_prompt(url, question, collection_name, client):
    collection = create_embedding(url, collection_name, client)
    relevant_chunks = collection.query(query_texts=[question], n_results=5)
    context = "\n\n\n".join(relevant_chunks["documents"][0])
    return (
        f"<|begin_of_text|>\n"
        f"<|start_header_id|>system<|end_header_id|>\n"
        f"You are a helpful AI assistant.\n"
        f"<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"### Context:\n{context}\n\n"
        f"### Instruction:\n"
        f"Answer the question concisely. Respond with 'unanswerable' if no context matches.\n\n"
        f"### Question:\n{question}\n"
        f"<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )


def get_model(model_type, max_tokens, min_tokens, decoding, temperature, top_k, top_p):
    params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature,
        GenParams.TOP_K: top_k,
        GenParams.TOP_P: top_p
    }
    model = Model(
        model_id=model_type,
        params=params,
        credentials={"apikey": st.session_state.api_key, "url": st.session_state.watsonx_url},
        project_id=st.session_state.watsonx_project_id
    )
    return model


def answer_questions_from_web(api_key, project_id, watsonx_url, url, question, collection_name, client):
    st.session_state.api_key = api_key
    st.session_state.watsonx_project_id = project_id
    st.session_state.watsonx_url = watsonx_url

    model = get_model("meta-llama/llama-3-70b-instruct", 100, 50, DecodingMethods.GREEDY, 0.7, 50, 1)
    prompt = create_prompt(url, question, collection_name, client)
    response = model.generate(prompt=prompt)
    return response['results'][0]['generated_text'].strip()


def main():
    if 'watsonx_project_id' not in st.session_state:
        st.session_state['watsonx_project_id'] = ""
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ""
    if 'watsonx_url' not in st.session_state:
        st.session_state['watsonx_url'] = default_url

    st.title("IBM watsonx.ai - RAG Web Demo")
    st.sidebar.header("Settings")
    api_key = st.sidebar.text_input("API Key", st.session_state.api_key, type="password")
    project_id = st.sidebar.text_input("Project ID", st.session_state.watsonx_project_id)
    watsonx_url = st.sidebar.text_input("Watsonx URL", st.session_state.watsonx_url)

    if api_key: st.session_state.api_key = api_key
    if project_id: st.session_state.watsonx_project_id = project_id
    if watsonx_url: st.session_state.watsonx_url = watsonx_url

    user_url = st.text_input("Provide a URL")
    question = st.text_area("Question", height=100)
    client = chromadb_client()
    collection_name = "base"

    if st.button("Answer the question"):
        if st.session_state.api_key and st.session_state.watsonx_project_id and st.session_state.watsonx_url and user_url:
            response = answer_questions_from_web(api_key, project_id, watsonx_url, user_url, question, collection_name, client)
            st.subheader("Response")
            st.write(response)
        else:
            st.warning("Please provide all credentials in the sidebar.")

    if st.sidebar.button("Clean Memory"):
        clear_collection(collection_name, client)


if __name__ == "__main__":
    main()
