import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from ingestion import Ingestion

class AttributeInfo(BaseModel):
    name: str
    description: str
    type: str

class Retriever:
    def __init__(self,vector_db,langchain_documents):
        if vector_db is not None:
            self.vector_retriever=vector_db.as_retriever(search_kwargs={"k":5})
            self.llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)
            # Instead of a list of strings, use a list of dictionaries detailing the metadata
            metadata_field_info = [
        {
            "name": "doc_category",
            "description": "The category of the document, such as 'finance' for SEC filings or 'general_doc'.",
            "type": "string",
        },
        {
            "name": "ticker",
            "description": "The stock ticker symbol of the company.",
            "type": "string",
        },
        {
            "name": "year",
            "description": "The year the document was filed.",
            "type": "integer",
        }
    ]

            self.vector_retriever = SelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=vector_db,
                document_contents="Detailed financial documents and SEC filings.",
                metadata_field_info=metadata_field_info,
    )
            self.bm25_retriever=BM25Retriever.from_documents(langchain_documents)
            self.bm25_retriever.k = 5
            self.hybrid_retriever=EnsembleRetriever(
                retrievers=[self.vector_retriever,self.bm25_retriever],
                weights=[0.7,0.3]
            )
            cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
            self.master_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=self.hybrid_retriever
            )
        else:
            self.master_retriever = None 
            print("Retriever initialized without vector_db.")
    def search(self, user_query):
        if self.master_retriever is None:
            return ["Error: Vector Database not connected."]
        return self.master_retriever.invoke(user_query)
    
