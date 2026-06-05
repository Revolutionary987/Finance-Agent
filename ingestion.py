import os
from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()

import re
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.chunking.title import chunk_by_title
from langchain_postgres import PGVector
from unstructured.partition.html import partition_html

class Ingestion:
    def __init__ (self, file_path):
        self.docs=file_path
        self.chunks=[]
        self.elements=[]
        self.metadata = {}
        self.model=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0,api_key=os.getenv("GROQ_API_KEY"))

    @traceable(name="Partitioning")
    async def partition(self):
        if not os.path.exists(self.docs):
            raise FileNotFoundError("Couldn't find the file")
        html_strings=await self.extract_html(self.docs)
        self.elements=partition_html(
            text=html_strings,
        )
        return self.elements
    async def extract_metadata(self,raw_text):
        metadata={
            "Company_Name":"Unknown",
            "Year":"Unknown",
            "Filed_date":"Unknown",
            "Expired_date":"Unknown",
            "Doc_type":"Unknown"
        }
        # refer re library documentation
        # \s* - Neglect or skip the whitespaces \s - Whitespaces , * - Zero or more times
        # () - Tells python to grab the text and save it also return as a variable
        # .*- Tells to grab everything 
        company_name=re.search(r"COMPANY CONFORMED NAME:\s*(.*)",raw_text)
        if company_name:
        # group(0) means returns entire match
        # EX: COMPANY COMFORMED NAME  APPLE
        # group(1) means return specific item present in the parentesis
        # APPLE

            metadata["Company_Name"]=company_name.group(1).strip()
        # \d{number} means grab the exact num of elementws specified
        published_yr=re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})",raw_text)
        if published_yr:
            raw_date=published_yr.group(1).strip()
            metadata["Expired_date"]=f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            metadata["Year"]=raw_date[:4]
        filed_match = re.search(r"FILED AS OF DATE:\s*(\d{8})",raw_text)
        if filed_match:
            raw_filed = filed_match.group(1).strip()
            # Format "20241101" -> "2024-11-01"
            metadata["Filed_date"] = f"{raw_filed[:4]}-{raw_filed[4:6]}-{raw_filed[6:]}"
        doc_type=re.search(r"CONFORMED SUBMISSION TYPE:\s*(.*)",raw_text)
        if doc_type:
            metadata["Doc_type"]=doc_type.group(1).strip()
        return metadata
    
    @traceable(name="Chunking")
    async def chunking(self):
        if len(self.elements) == 0:
            raise ValueError("No elements found to chunk.")
        
        data=self.metadata
        # sees the generated headings like title , tables etc from the partition_html then chunks them by title\
        chunks=chunk_by_title(
            elements=self.elements,
            max_characters=2000,
            combine_text_under_n_chars=500,
            overlap=300
        )

        for chunk in chunks:
            if "Table" in str(type(chunk)) and hasattr(chunk.metadata,"text_as_html") and chunk.metadata.text_as_html:
                page_content=chunk.metadata.text_as_html
                chunk_type="table"
            else:
                page_content=chunk.text
                chunk_type="text"

            doc=Document(
                page_content=page_content,
                metadata={
                    "company": data["Company_Name"],
                    "document_type": data["Doc_type"],
                    "financial_period_end":data["Expired_date"],
                    "legally_filed_date":data["Filed_date"],
                    "type": chunk_type,
                    "section": chunk.metadata.title if hasattr(chunk.metadata, "title") else "Unknown"
            }
        )
            self.chunks.append(doc)
        return self.chunks
# it does remove html tags and then converts the content into texts like Title,Table etc on seeing the html tag
    async def extract_html(self,file_path):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_file = f.read()
        self.metadata=await self.extract_metadata(raw_file)
        documents = re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', raw_file, re.DOTALL | re.IGNORECASE)
        for doc in documents:
            if re.search(r'<TYPE>(10-K|10-Q)', doc, re.IGNORECASE):
                # The iXBRL safe regex to capture the HTML block
                html_match = re.search(r'(<html[^>]*>.*?</html>)', doc, re.DOTALL | re.IGNORECASE)
                if html_match:
                    print("[SYSTEM] 10-K Found! Stripping Base64 images and XML noise...")
                    return html_match.group(1)
                    
        raise ValueError("Could not find a valid 10-K HTML block in this file.")
    
    @traceable(name="embeddings")
    async def embedding(self):
        model_name="BAAI/bge-m3"
        model_kwargs={"device":"cpu"}
        encode_kwargs={"normalize_embeddings":True}
        embedding_model=HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        raw_url = os.getenv("DATABASE_URL")
        RENDER_DB_URL = raw_url.replace("postgres://", "postgresql+psycopg://")
        if raw_url.startswith("postgresql://"):
            RENDER_DB_URL = raw_url.replace("postgresql://", "postgresql+psycopg://")
        elif raw_url.startswith("postgres://"):
            RENDER_DB_URL = raw_url.replace("postgres://", "postgresql+psycopg://")
        else:
            RENDER_DB_URL = raw_url
        vector_db = PGVector(
                embeddings=embedding_model,
                collection_name="aegis_db",
                connection=RENDER_DB_URL,
                use_jsonb=True,
                async_mode=True, 
        )
        if self.chunks:
            await vector_db.aadd_documents(self.chunks)     
            return vector_db
