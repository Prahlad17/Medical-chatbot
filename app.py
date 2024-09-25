from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.chains import RetrievalQA
import subprocess

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

embeddings = download_hugging_face_embedding()

pc = Pinecone(api_key=PINECONE_API_KEY)
index1 = pc.Index("medical-chatbot1")

#Loading the index
vectorstore = PineconeVectorStore(index_name="medical-chatbot1", embedding=embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])


chain_type_kwargs={"prompt": PROMPT}



# Function to download the model if it's not already present
def download_model():
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin"
    model_dir = "model/"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "llama-2-7b-chat.ggmlv3.q4_0.bin")

    if not os.path.exists(model_path):
        print("Downloading model...")
        subprocess.run(["wget", "--no-cache", model_url, "-O", model_path])
        print(f"Downloaded model to {model_path}")
    else:
        print("Model already exists. Skipping download.")

# Download model before proceeding
download_model()

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})





# def query_index(query, top_k=3):
#     query_embedding = embeddings.embed_query(query)
    
#     results = index1.query(
#         vector=query_embedding,
#         top_k=top_k,
#         include_metadata=True
#     )
    
#     return [match['metadata']['text'] for match in results['matches']]

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)



@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        if request.is_json:
            msg = request.json.get('message')
        else:
            msg = request.form.get('msg')
        
        if not msg:
            return jsonify({"error": "No message provided"}), 400
        
        input = msg
        print(input)
        result = qa({"query": input})
        print("Response : ", result["result"])
        return str(result["result"])
    else:
        return jsonify({"error": "Method not allowed"}), 405
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)