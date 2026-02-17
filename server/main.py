from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from modules.load_vectostore import load_vectorstore
from logger import logger
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone   
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()



app = FastAPI(title="RagBot2.0")
PK_API_KEY = os.environ.get("PINECONE_API_KEY")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 1. Pinecone + Embedding setup
pc = Pinecone(api_key=PK_API_KEY)
index = pc.Index("medical-index")
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = PineconeVectorStore( index=index,embedding=embed_model )
retriever = vectorstore.as_retriever()



@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})



@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    try:
        logger.info(f"recived {len(files)} files")
        load_vectorstore(files)
        logger.info(f"documents loaded into vectorstore")
        return {"message": f"Successfully uploaded {len(files)} files."}
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "Failed to upload files."})



@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"Received question: {question}")

        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)
  
        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})

        


@app.get("/test")
async def test():
    return {"message": "API testing is successful!"}

















