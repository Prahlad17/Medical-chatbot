from src.helper import load_pdf, text_split, download_hugging_face_embedding
from pinecone import Pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embedding()


# initialize connection to pinecone (get API key at app.pinecone.io)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("medical-chatbot1")

from tqdm.auto import tqdm

batch_size = 256

for i in tqdm(range(0, len(text_chunks), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(text_chunks))
    # create batch of texts
    texts = [t.page_content for t in text_chunks[i:i_end]]
    # create embeddings
    embeds = embeddings.embed_documents(texts)
    # create metadata
    meta = [{"text": t} for t in texts]
    # create unique IDs
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, embeds, meta))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)