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
from langchain_core.documents import Document

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
                    "description": "The name of the company (e.g., 'Apple Inc.'). STRICT RULE: Use 'eq' or 'like'. Do NOT use 'contain'.",
                    "type": "string",
                },
                {
                    "name": "document_type",
                    "description": "The type of SEC filing (e.g., '10-K', '10-Q'). STRICT RULE: Use 'eq' or 'like'.",
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
                    "description": "The specific SEC document section title (e.g., 'Risk Factors', 'Item 8'). STRICT RULE: Use 'like' for partial matches. Do NOT use 'contain'.",
                    "type": "string",
                }
            ]
            
            strict_document_prompt = (
                "Detailed financial documents and SEC filings. "
                "CRITICAL INSTRUCTION: When creating metadata filters for string fields, "
                "you are STRICTLY FORBIDDEN from using the 'contain' operator. "
                "You MUST use the 'like' or 'eq' operator instead. "
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
        hierarchical_docs = []
        # To get parent content from the chunking 
        for doc in raw_docs:
            parent_context = doc.metadata.get("parent_context")
            # we are creating langchain docs so we can get send the parent chunk to the llm 
            # when retriever is invoked langchain gets child chunks in langchain docs format but we need to pass the parent chunks so we loop through it find the parent content and create the langchain docs of the parent chunk 
            if parent_context:
                expanded_doc = Document(
                    page_content=parent_context,
                    metadata=doc.metadata
                )
                hierarchical_docs.append(expanded_doc)
            else:
                hierarchical_docs.append(doc)
                
        # Ensuring that there aren't identical parent pages multiple times 
        seen_contents = set()
        unique_docs = []
        for doc in hierarchical_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_docs.append(doc)
                
        return {"documents": unique_docs}

    
