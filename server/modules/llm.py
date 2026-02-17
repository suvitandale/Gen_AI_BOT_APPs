from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()
GK_TOKEN = os.environ.get("GROQ_API_KEY")
PK_API_KEY = os.environ.get("PINECONE_API_KEY")

def get_llm_chain(retriever):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=GK_TOKEN,
    )


    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are **MediBot**, an AI-powered assistant trained to help users understand medical documents and health-related questions.
        Your job is to provide clear, accurate, and helpful responses based **only on the provided context**.
        ---
        🔍 **Context**:
        {context}

        🙋‍♂️ **User Question**:
        {question}

        ---
        💬 **Answer**:
        - Respond in a calm, factual, and respectful tone.
        - Use simple explanations when needed.
        - If the context does not contain the answer, say: "I'm sorry, but I couldn't find relevant information in the provided documents."
        - Do NOT make up facts.
        - Do NOT give medical advice or diagnoses.
        """)
    

    rag_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | prompt
    | llm
    | StrOutputParser()
)
    
    return rag_chain