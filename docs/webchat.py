# For reading credentials from the .env file
import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from chromadb.api.types import EmbeddingFunction

# WML python SDK
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

import requests
from bs4 import BeautifulSoup
import spacy
import chromadb
import en_core_web_md


# Important: hardcoding the API key in Python code is not a best practice. We are using
# this approach for the ease of demo setup. In a production application these variables
# can be stored in an .env or a properties file

# URL of the hosted LLMs is hardcoded because at this time all LLMs share the same endpoint
url = "https://us-south.ml.cloud.ibm.com"

# These global variables will be updated in get_credentials() function
watsonx_project_id = ""
# Replace with your IBM Cloud key
api_key = ""

def get_credentials():

    load_dotenv()
    # Update the global variables that will be used for authentication in another function
    globals()["api_key"] = os.getenv("api_key", None)
    globals()["watsonx_project_id"] = os.getenv("project_id", None)

# The get_model function creates an LLM model object with the specified parameters

def get_model(model_type, max_tokens, min_tokens, decoding, temperature, top_k, top_p):
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature,
        GenParams.TOP_K: top_k,
        GenParams.TOP_P: top_p,
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
    )

    return model

def get_model_test(model_type, max_tokens, min_tokens, decoding, temperature):
    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding,
        GenParams.TEMPERATURE: temperature
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": api_key,
            "url": url
        },
        project_id=watsonx_project_id
    )

    return model

# Set up cache directory (consider user-defined location)
current_dir = os.getcwd()
cache_dir = os.path.join(current_dir, ".cache")
# Create cache directory if necessary
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
# Set the Hugging Face cache directory
os.environ["HF_HOME"] = cache_dir    
# Download the model (specify the correct model identifier)
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
#model_name = "all-MiniLM-L6-v2"  
model = SentenceTransformer(model_name, cache_folder=cache_dir)
# Print confirmation message
print(f"Model '{model_name}' downloaded and loaded from cache directory: {cache_dir}")

# Embedding function
class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
    MODEL = model
    def __call__(self, texts):
        return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()

def extract_text(url):
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract contents of <p> elements
            p_contents = [p.get_text() for p in soup.find_all('p')]

            # Print the contents of <p> elements
            print("\nContents of <p> elements: \n")
            for content in p_contents:
                print(content)
            raw_web_text = " ".join(p_contents)
            # remove \xa0 which is used in html to avoid words break acorss lines.
            cleaned_text = raw_web_text.replace("\xa0", " ")
            return cleaned_text
        else:
            print(f"Failed to retrieve the page. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def split_text_into_sentences(text):
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    cleaned_sentences = [s.strip() for s in sentences]
    return cleaned_sentences

def create_embedding(url, collection_name,client):
    cleaned_text = extract_text(url)
    cleaned_sentences = split_text_into_sentences(cleaned_text)
    collection = client.get_or_create_collection(collection_name)
    # Upload text to chroma
    collection.upsert(
        documents=cleaned_sentences,
        metadatas=[{"source": str(i)} for i in range(len(cleaned_sentences))],
        ids=[str(i) for i in range(len(cleaned_sentences))],
    )

    return collection


def create_prompt_old(url, question, collection_name, client):
    # Create embeddings for the text file
    collection = create_embedding(url, collection_name, client)

    # query relevant information
    relevant_chunks = collection.query(
        query_texts=[question],
        n_results=5,
    )
    context = "\n\n\n".join(relevant_chunks["documents"][0])
    # Please note that this is a generic format. You can change this format to be specific to llama
    prompt = (f"{context}\n\nPlease answer the following question in one sentence using this "
              + f"text. "
              + f"If the question is unanswerable, say \"unanswerable\". Do not include information that's not relevant to the question."
              + f"Question: {question}")

    return prompt

def create_prompt(url, question, collection_name,client):
  try:
    # Create embeddings for the text file
    collection = create_embedding(url, collection_name,client)
  except Exception as e:
    return f"Error creating embeddings: {e}"
  
  try:
    # Query relevant information
    relevant_chunks = collection.query(
        query_texts=[question],
        n_results=5,
    )
    context = "\n\n\n".join(relevant_chunks["documents"][0])
  except Exception as e:
    return f"Error querying the collection: {e}"
  
  # Create the prompt
  prompt = (
      "<|begin_of_text|>\n"
      "<|start_header_id|>system<|end_header_id|>\n"
      "You are a helpful AI assistant.\n"
      "<|eot_id|>\n"
      "<|start_header_id|>user<|end_header_id|>\n"
      f"### Context:\n{context}\n\n"
      f"### Instruction:\n"
      f"Please answer the following question based on the above context. Your answer should be concise and directly address the question. "
      f"If the question is unanswerable based on the given context, respond with 'unanswerable'.\n\n"
      f"### Question:\n{question}\n"
      "<|eot_id|>\n"
      "<|start_header_id|>assistant<|end_header_id|>\n"
  )

  return prompt



def main():

    # Get the API key and project id and update global variables
    get_credentials()
   
    # Try diffrent URLs and questions
    url = "https://huggingface.co/learn/nlp-course/chapter1/2?fw=pt"

    question = "What is NLP?"
    collection_name = "test_web_RAG"

    answer_questions_from_web(api_key, watsonx_project_id, url, question, collection_name)


def answer_questions_from_web(request_api_key, request_project_id, url, question, collection_name,client):
    # Update the global variable
    globals()["api_key"] = request_api_key
    globals()["watsonx_project_id"] = request_project_id

    # Specify model parameters
    model_type = "meta-llama/llama-2-70b-chat"
    #model_type = "meta-llama/llama-3-70b-instruct"
    max_tokens = 100
    min_tokens = 50
    top_k = 50
    top_p = 1
    decoding = DecodingMethods.GREEDY
    temperature = 0.7

    # Get the watsonx model = try both options
    model = get_model(model_type, max_tokens, min_tokens, decoding, temperature, top_k, top_p)
    # Get client Chromadb
    client = chromadb.Client()

    # Get the prompt
    complete_prompt = create_prompt(url, question, collection_name,client)

    # Let's review the prompt
    print("----------------------------------------------------------------------------------------------------")
    print("*** Prompt:" + complete_prompt + "***")
    print("----------------------------------------------------------------------------------------------------")

    generated_response = model.generate(prompt=complete_prompt)
    response_text = generated_response['results'][0]['generated_text']

    # Remove trailing white spaces
    response_text = response_text.strip()

    # print model response
    print("--------------------------------- Generated response -----------------------------------")
    print(response_text)
    print("*********************************************************************************************")

    return response_text

# Invoke the main function
if __name__ == "__main__":
    main()
