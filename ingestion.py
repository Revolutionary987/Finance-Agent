import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
load_dotenv()
import tempfile
import uuid
import re
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter

class Ingestion:
    def __init__ (self, file_path):
        self.docs=file_path
        self.chunks=[]
        self.elements=[]
        self.metadata = {}
        self.MODEL_ID="BAAI/bge-m3"
        self.tokenizer=AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model=ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0,api_key=os.getenv("GROQ_API_KEY"))
        self.child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer,
            chunk_size=256,
            chunk_overlap=32,
        )
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
        company_name=re.search(r"COMPANY CONFORMED NAME:\s*(.*)",raw_text,re.IGNORECASE)
        if company_name:
        # group(0) means returns entire match
        # EX: COMPANY COMFORMED NAME  APPLE
        # group(1) means return specific item present in the parentesis
        # APPLE

            metadata["Company_Name"]=company_name.group(1).strip()
        # \d{number} means grab the exact num of elementws specified
        published_yr=re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})",raw_text,re.IGNORECASE)
        if published_yr:
            raw_date=published_yr.group(1).strip()
            metadata["Expired_date"]=f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            metadata["Year"]=raw_date[:4]
        filed_match = re.search(r"FILED AS OF DATE:\s*(\d{8})",raw_text,re.IGNORECASE)
        if filed_match:
            raw_filed = filed_match.group(1).strip()
            # Format "20241101" -> "2024-11-01"
            metadata["Filed_date"] = f"{raw_filed[:4]}-{raw_filed[4:6]}-{raw_filed[6:]}"
        doc_type=re.search(r"CONFORMED SUBMISSION TYPE:\s*(.*)",raw_text,re.IGNORECASE)
        if doc_type:
            metadata["Doc_type"]=doc_type.group(1).strip()
        return metadata
    # @staticmethod means this function is a utility helper. It doesn't modify or read any instance variables inside your main class.It just takes a chunk, processes it, and returns a result.
    # chunk is a raw DocChunk object yielded by Docling's HybridChunker.
    @staticmethod
    def _doc_item_refs(chunk):
        # It loops through every chunk produced by docling. For every element, it grabs its unique address path (item.self_ref) and packs it into a Python set.
        return {item.self_ref for item in chunk.meta.doc_items}
# it does remove html tags and then converts the content into texts like Title,Table etc on seeing the html tag
    async def extract_html_and_meta(self,file_path):
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
    @traceable(name="Partitioning")
    async def partition(self):
        # Isolate the HTML data string and populate self.metadata tracking dictionary
        html_content = await self.extract_html_and_meta(self.docs)
        
        # DoclingLoader requires a physical file path. We use a NamedTemporaryFile to stream the text securely without leaving permanent garbage on the filesystem.
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(html_content)
            temp_file_path = temp_file.name

        try:
            print("Docling loader")
            # We set max_tokens=2048 using our model's real tokenizer rules (not raw characters) 
            # to capture complete financial sections, contextual descriptions, and full tables.
            # Loader converts docs, python lists ,tuples etc into langchain documents do the llm or embedding models can use
            # why Doclingloader as it the best for financial rag and it uses hybridchunker instead of unstructured which uses partition_html
            # why converting to langchain docs again ??
            # see we get output in langchain docs only but the docs will be in this format {"source":"some_random.html"} that html will be deleted 
            # so we won't get the detailed metadata if we wanna pass the content only we can load the loader but to get the metadata we should create langchain docs again with the content from here and attach the metadata
            # also LangChain Document metadata fields are completely immutable (read-only)
            converter = DocumentConverter()
            dl_doc = converter.convert(temp_file_path).document
            parent_chunker = HybridChunker(tokenizer=self.tokenizer, max_tokens=2048)
            parent_chunks = list(parent_chunker.chunk(dl_doc))
            child_chunks = None
            try:
                child_chunker = HybridChunker(tokenizer=self.tokenizer, max_tokens=256)
                # chunker.chunk returns a generator obj for easier looping and debugging we make it as a list
                child_chunks = list(child_chunker.chunk(dl_doc))
            except Exception as e:
                print(f"falling back to RecursiveCharacterTextSplitter for this document: {e}")
            for p_chunk in parent_chunks:
                parent_id = str(uuid.uuid4())
                # contextualize( )converts chunks into a standard Markdown string.
                parent_text = parent_chunker.contextualize(p_chunk)
 
                segments = []
                if child_chunks is not None:
                    # this lines stores all the parent ids we created into parent_refs
                    parent_refs = self._doc_item_refs(p_chunk)
                    # checks whether the child ids are subset of parent ids so that we can map the child chunks exactly to the parent chunks 
                    matching_children = [c for c in child_chunks if self._doc_item_refs(c) <= parent_refs]
                    if matching_children:
                        segments = [child_chunker.contextualize(c) for c in matching_children]
                    else:
                        print("no child chunks falling back to text splitter for this parent.")
                        segments = self.child_splitter.split_text(parent_text)
                else:
                    segments = self.child_splitter.split_text(parent_text)
 
                # checking for markdown tables
                for segment in segments:
                    chunk_type = "table" if "|" in segment and "---" in segment else "text"
                
                    doc = Document(
                            page_content=segment,
                            metadata={
                                "parent_id": parent_id,
                                "parent_context": parent_text,
                                "company": self.metadata.get("Company_Name", "Unknown"),
                                "document_type": self.metadata.get("Doc_type", "Unknown"),
                                "financial_period_end": self.metadata.get("Expired_date", "Unknown"),
                                "legally_filed_date": self.metadata.get("Filed_date", "Unknown"),
                                "type": chunk_type,
                                "section": "Unknown"
                            }
                        )
                    self.chunks.append(doc)
                
        finally:
            # Cleaning up filesystem context immediately upon completion
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
        return self.chunks
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
        if not raw_url:
            raise ValueError("DATABASE_URL environment variable is missing!")
        ASYNC_DB_URL = re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", raw_url.strip())
        vector_db = PGVector(
                embeddings=embedding_model,
                collection_name="aegis_db",
                connection=ASYNC_DB_URL,
                use_jsonb=True,
                async_mode=True, 
        )
        if not self.chunks:
            raise FileNotFoundError("Couldn't find the chunks")
        batch_size = 100
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            try:
            # Upload a single controlled batch block
                await vector_db.aadd_documents(batch)
            except Exception as e:
                raise e   
        return vector_db
