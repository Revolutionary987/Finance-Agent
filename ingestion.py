import os
import json
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from pydantic import BaseModel
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import SelfQueryRetriever
from typing import List,Dict
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title


class Ingestion:
    def __init__ (self, docs):
        self.docs=docs
        self.chunks=[]
        self.elements=[]
        self.model=ChatGroq(model="llama3-70b-8192", temperature=0,api_key=os.getenv("GROQ_API_KEY"))
    def partition(self):
        if not os.path.exists(self.docs):
            raise FileNotFoundError("Couldn't find the file")
        
        self.elements=partition(
            filename=self.docs,
            extract_images_in_pdf=True,
            strategy="hi_res",
            infer_table_structure=True,
            chunking_strategy="by_title",
            extract_image_block_to_payload=True
        )
        return self.elements
    def chunkdocs(self):
        if len(self.elements)==0:
            raise ValueError("No elements found")
        self.chunks=chunk_by_title(
            elements=self.elements,
            max_characters=2000,
            overlap=300,
            include_orig_elements=True
        )
        return self.chunks
    
    def sep_contents(self,chunk):
        content={
            "text":[],
            "images":[],
            "tables":[],
            "types":[],
        }
        if hasattr(chunk,"text") and chunk.text:
            content["text"].append(chunk.text)
        if hasattr(chunk,"metadata") and hasattr(chunk.metadata,"orig_elements"):
            for element in chunk.metadata.orig_elements:
                ele_type=(type(element).__name__).title()

                if ele_type=="Table":
                    content["types"].append("Table")
                    html_table=getattr(chunk.metadata,"table_as_html",element.text)
                    content["tables"].append(html_table)
                if ele_type=="Image":
                    content["types"].append("Image")
                    content["images"].append(element.metadata.image_base64)
        content["types"]=list(set(content["types"]))
        return content
    def summary(self,text:str,tables:list[str],images:list[str])->str:
        try:
            prompt=f"""You are an expert technical assistant. Analyze the following content 
            and generate a highly detailed, searchable summary.
            Include key facts, metrics, and describe any visual anomalies.
            
            TEXT:
            {text}
            
            TABLES:
            {tables}

            """
            message_content = [
                {"type": "text", "text": prompt}
            ]
            # Llm talks to json not binary files 
            for img_base64 in images:
                message_content.append({
                    "type":"image_url",
                    "image_url":{"url":f"data:image/jpeg;base64,{img_base64}"}
                })
            message=HumanMessage(content=message_content)
            response=self.model.invoke([message])
            return response.content
        except Exception as e:
            return (f"Summary failed due to {e}")

    def document(self):
        langchain_documents=[]
        if len(self.chunks)==0:
            raise ValueError("No data found")
        for chunk in self.chunks:
            data=self.sep_contents(chunk)

            summary_gen=self.summary(
                text=data["text"],
                tables=data["tables"],
                images=data["images"],
            )
            docs=Document(
                page_content=summary_gen,
                metadata={
                    "original_data":json.dumps(
                        {
                            'raw_text':data['text'],
                            'table_as_html':data['tables'],
                            'base_64_image':data['images'],
                        }
                    )
                } 
            )
            langchain_documents.append(docs)
        return langchain_documents
    
    def embedding(self,langchain_documents,persist_directory="ddb/chroma_db"):
        model_name="BAAI/bge-m3"
        model_kwargs={"device":"cpu"}
        encode_kwargs={"normalize_embeddings":True}
        embedding_model=HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        vector_db=Chroma.from_documents(
            documents=langchain_documents,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        return vector_db
