import os
import json
from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()

from pydantic import BaseModel
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import SelfQueryRetriever
from langchain_groq import ChatGroq
from langchain_community.retrievers import BM25Retriever
from langchain_core.structured_query import Comparator

class AttributeInfo(BaseModel):
    name: str
    description: str
    type: str

pgvector_comparators = [
    Comparator.EQ,
    Comparator.NE,
    Comparator.GT,
    Comparator.GTE,
    Comparator.LT,
    Comparator.LTE,
    Comparator.IN,
    Comparator.NIN,
    Comparator.LIKE 
]
class Retriever:
    def __init__(self,vector_db,langchain_documents):
        if vector_db is not None:

            self.vector_retriever=vector_db.as_retriever(search_kwargs={"k":5})
            self.llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0,api_key=os.getenv("GROQ_API_KEY"))

            # Instead of a list of strings, use a list of dictionaries detailing the metadata
            metadata_field_info = [
                {
                    "name": "company",
                    "description": "The name of the company (e.g., 'Apple Inc.'). STRICT RULE: Use 'eq' or 'ilike'. Do NOT use 'contain'.",
                    "type": "string",
                },
                {
                    "name": "document_type",
                    "description": "The type of SEC filing (e.g., '10-K', '10-Q'). STRICT RULE: Use 'eq' or 'ilike'.",
                    "type": "string",
                },
                {
                    "name": "financial_period_end",
                    "description": "The date the financial quarter ended, in YYYY-MM-DD format. You can use 'gt', 'lt', or 'eq' to filter date ranges.",
                    "type": "string",
                },
                {
                    "name": "legally_filed_date",
                    "description": "The date the document was legally filed to the public, in YYYY-MM-DD format. You can use 'gt', 'lt', or 'eq' to filter date ranges.",
                    "type": "string",
                },
                {
                    "name": "type",
                    "description": "The structure of the data. Strictly either 'text' for paragraphs or 'table' for financial grids.",
                    "type": "string",
                },
                {
                    "name": "section",
                    "description": "The specific SEC document section title (e.g., 'Risk Factors', 'Item 8'). STRICT RULE: Use 'ilike' for partial matches. Do NOT use 'contain'.",
                    "type": "string",
                }
            ]
            strict_document_prompt = (
                "Detailed financial documents and SEC filings. "
                "CRITICAL INSTRUCTION: When creating metadata filters for string fields, "
                "you are STRICTLY FORBIDDEN from using the 'contain' operator. "
                "You MUST use the 'ilike' or 'eq' operator instead. "
                "Failure to follow this rule will crash the SQL database."
            )

            self.vector_retriever = SelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=vector_db,
                document_contents=strict_document_prompt,
                metadata_field_info=metadata_field_info,
                allowed_comparators=pgvector_comparators,
    )
            if langchain_documents and len(langchain_documents) > 0:
                self.bm25_retriever = BM25Retriever.from_documents(langchain_documents)
                self.bm25_retriever.k = 3
                
                self.base_retriever = EnsembleRetriever(
                    retrievers=[self.vector_retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]
                )
            else:
                self.base_retriever = self.vector_retriever
            
            cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)
            self.master_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=self.base_retriever
            )
        else:
            self.master_retriever = None 
            print("Retriever initialized without vector_db.")
    @traceable(name="retriving")
    async def search(self, user_query,config=None):
        if self.master_retriever is None:
            print("Error: Vector Database not connected.")
            return {"documents":[]}
        raw_docs = await self.master_retriever.ainvoke(user_query, config=config)
        return {"documents":raw_docs}
    
