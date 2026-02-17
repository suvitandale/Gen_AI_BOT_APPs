import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medical-index"

UPLOAD_DIR = "./data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone (Pinecone is a vector database service)
pinecone = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it doesn't exist (Index means vector database/table)
existing_indexs = [index.name for index in pinecone.list_indexes()]

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if PINECONE_INDEX_NAME not in existing_indexs:
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

    pinecone.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="dotproduct", spec=spec)
    print(f"Created Pinecone index: {PINECONE_INDEX_NAME}")

    while not pinecone.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pinecone.Index(PINECONE_INDEX_NAME)



# UpLoad, split, embed and upsert PDF content
def load_vectorstore(uploaded_files):

    file_paths = []
    for file in uploaded_files:
        saved_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(saved_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(saved_path)

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [ {
        **chunk.metadata,
        "text": chunk.page_content     } for chunk in chunks]

        ids = [
            f"{Path(file_path).stem}_{i}"
            for i in range(len(chunks))
        ]

        print(f"Embedding {len(texts)} chunks...")


        embeddings = embed_model.embed_documents(texts)
        print("Uploading to Pinecone...")

        batch_size = 100

        with tqdm(total=len(embeddings)) as pbar:

            for i in range(0, len(embeddings), batch_size):

                batch = list(zip(
                    ids[i:i+batch_size],
                    embeddings[i:i+batch_size],
                    metadatas[i:i+batch_size]
                ))

                index.upsert(vectors=batch)

                pbar.update(len(batch))


        print(f"Upload complete: {file_path}")