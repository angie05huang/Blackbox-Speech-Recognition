# === Imports=== 
import time
import ollama
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.ollama import OllamaEmbeddings  
import os
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from uuid import uuid4
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader

# === Load Environment Variables ===
load_dotenv() # Load variables from .env file into the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Retrieve OpenAI API key

# === Configuration ===
DB_CHROMA_PATH = "./vectorstore"

# === Embedding Function Setup ===
# Creates a function using OpenAI's embedding model, authenticated via API key
embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
def chroma_ingest(file_path, vector_dir):
    """
    Ingests a PDF file into a Chroma vectorstore after embedding its text content.

    Parameters:
        file_path (str): Path to the uploaded PDF file
        vector_dir (str): Directory to store the vectorstore for a specific project
    """
    
    # Checks if uploaded file is pdf
    if not file_path.lower().endswith('.pdf'):
        print("Not a PDF file")
        return

    # Checks if the file is empty
    if os.path.getsize(file_path) == 0:
        print("Saved file is 0 bytes. Aborting.")
        return  
    
    # Load the PDF file using LangChain's PyPDFLoader
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        print(f"PDF load failed: {e}")
        os.remove(file_path)
        return
    
    # Initialize Chroma vectorstore for the specified directory
    vectorstore = Chroma(persist_directory=vector_dir, embedding_function=embedding_function)

    # Ingest each page of the PDF into the vectorstore
    count = 1
    for doc in docs:
        new_doc = Document(page_content = doc.page_content + "\n\n"+os.path.basename(file_path), metadata ={"source": os.path.basename(file_path)+", Page "+ str(count)} )
        vectorstore.add_documents(documents=[new_doc], ids=[str(uuid4())])
        count+=1
    # Delete the original file after ingestion
    os.remove(file_path)

