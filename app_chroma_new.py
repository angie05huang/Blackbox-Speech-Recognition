# === Imports ===
import time
from ollama import Client
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings.ollama import OllamaEmbeddings  
import os
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import shutil
import time

#classifier = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M")
# [{'label': 'JAILBREAK', 'score': 0.9999452829360962}]

# Define model and initialize chat history

desired_model = "llama3"
#DB_CHROMA_PATH = "./vectorstore{project_number}"
#embedding_function = OllamaEmbeddings(model="llama3")  # change this based on your embeddings

# === Environment Setup ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding_fucntion turns documents into vectors
embedding_function = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# === Prompt Template ===
# Defines the structure the AI should follow when answering questions based on provided documents
prompt_template = """You are an Expert Research professional helping to synthesize key findings, analysis and summary.
Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
You only know what is contained in the provided context. Do not use any outside knowledge.
ALWAYS return a "SOURCES" part in every part of your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer. Please indicate the exact name of the document. 
Example of your response should be as follows:

Context: {context}
Question: {question}

Only return the helpful answer and sources below and nothing else. The sources section should reference the exact name of the document.
Helpful answer:
Sources:
"""
# Configures the prompt and the vectorstore
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Stores user/assistant message history to maintain conversation state
chat_history = []

def retrieve_bot_answer(user_input, project_number):
    """
    Handles the user's question by retrieving relevant document context using vector search,
    formatting it into a prompt, and sending it to the LLaMA3 model for response.

    Parameters:
        user_input (str): The user's question
        project_number (str or int): Project-specific vectorstore ID

    Returns:
        str: AI-generated answer including sources
    """
    global chat_history

    # Retrieve relevant context from project's vectorstore
    DB_CHROMA_PATH = f"./vector_store/vectorstore_{project_number}"

    # Vector Retrieval 
    start_time = time.time()
    vectorstore = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embedding_function)
    vector_time = time.time()

     # Search for top-k similar documents to the question
    docs = vectorstore.similarity_search(user_input, k=5)
    print("Retrieved docs and scores:", docs)
    cosine_time = time.time()
    if not docs: 
        return "can't find relavent context in the document"
    
    # Format the context from top-k documents
    context = "\n\n".join([f'{doc.page_content} [{doc.metadata}]' for doc in docs])

    # Fill in prompt template
    formatted_prompt = prompt.format(context=context, question=user_input)

    # Build and send chat
    chat_history.append({"role": "user", "content": formatted_prompt})


    
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    client = Client(host=ollama_host)

    response = client.chat(model="llama3", messages=chat_history)
    chat_time = time.time()
    
    # Extract and display response
    assistant_response = response["message"]["content"]
    print(f"Assistant:\n{assistant_response}")
    
    # Add assistant message to history
    chat_history.append({"role": "assistant", "content": assistant_response})
    total_time = time.time()

    # Keep chat history short: only last interaction (user + assistant)
    chat_history=[]
    print(vector_time- start_time, cosine_time-start_time, chat_time-start_time, total_time-start_time)
    return assistant_response

# === Utility: Delete Vectorstore ===
def delete_vector_store(project_number):
    """
    Deletes the entire vectorstore associated with the given project number.

    Parameters:
        project_number (str or int): Project-specific vectorstore identifier
    """
    DB_CHROMA_PATH = f"./vector_store/vectorstore_{project_number}"
    vectorstore = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embedding_function)
    vectorstore.delete_collection()
    
def get_all_files(project_number):
    DB_CHROMA_PATH = f"./vector_store/vectorstore_{project_number}"
    vectorstore = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embedding_function)
    files = vectorstore.get(include = ['metadatas'])["metadatas"]
    ans = set()
    
    for file in files:
        ans.add(file["source"].split(',')[0])
    print(ans)
    here = ""
    for file in ans:
        here+=file+'\n'
    here=here[:len(here)-2]
    return here