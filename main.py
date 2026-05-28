import json
import os
from retriever import Retriever
from ingestion import Ingestion
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.documents import Document

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
chat_history=[]
def memory(user_query):
    if chat_history:
        messages=[
            SystemMessage(content="Given chat history rewrite the user query as a standalone question and return the rewritten question")
        ]+chat_history+[HumanMessage(content=f"Question: {user_query}")]
    
        result=model.invoke(messages)
        question = result.content.strip()
    else:
        question = user_query
    return question
def chat(retrieve):
    while True:
        ques = input("\nEnter the query (or 'quit'): ")
        if 'quit' in ques.lower():
            break
    asked_ques=memory(ques)
    docs=retrieve.search(asked_ques)

    text=set()
    table=set()
    image=set()
    for doc in docs:
        meta = json.loads(doc.metadata.get("original_data","{}"))
        text.update(meta.get("raw_text",[]))
        table.update(meta.get("table_as_html", []))
        image.update(meta.get("base_64_image", []))
    text = "\n\n".join(text)
    table = "\n\n".join(table)

    llm_prompt = f"""You are an expert technical assistant, Your job is to provide accurate, professional answers based STRICTLY on the provided document context.

        ### INSTRUCTIONS:
        1. Review the provided context thoroughly.
        2. Answer the user's query using ONLY the information found in the context.
        3. If the context contains tables or image summaries, use them to enrich your answer.
        4. If the answer cannot be found in the context, you must output exactly: "I don't have enough information." Do not attempt to guess or use outside knowledge.

        ### CONTEXT:
        UNSTRUCTURED TEXT:
        {text}

        STRUCTURAL FINANCIAL TABLES (HTML):
        {table}

        ### USER QUERY:
        {asked_ques}
        """
    message_content = [
            {"type": "text", "text": llm_prompt}
        ]
    for img_b64 in image:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
            
    messages = [
            SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and the previous chat history"),
        ] + chat_history + [
            HumanMessage(content=message_content)
        ]
            
    result = model.invoke(messages)
    final_answer = result.content
    print(final_answer)
    chat_history.append(HumanMessage(content=ques))
    chat_history.append(AIMessage(content=final_answer))
    if len(chat_history) > 10:
        del chat_history[:2]

def get_exact_year(text_file_path):
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            # Read just the first 50 lines to find the header
            head = [next(f) for _ in range(50)]
            for line in head:
                if "FILED AS OF DATE:" in line:
                    # Extracts '2023' from '20230727'
                    return int(line.split(":")[1].strip()[:4]) 
    except Exception as e:
        print(f"[-] Error reading file header: {e}")
    return 2023 # Fallback year if something goes wrong


if __name__ == "__main__":
    # Point directly to the actual text file, not just the folder
    file_path = r"C:\Users\Tharun R Gowda\Desktop\financial rag\data\sec-edgar-filings\MSFT\10-K\0000950170-23-035122\full-submission.txt" 
    
    # Check if the text file exists and if the database hasn't been built yet
    if os.path.exists(file_path) and not os.path.exists("ddb/chroma_db"):
        print(f"\n[+] Found raw SEC filing. Extracting exact date...")
        
        # 1. Dynamically get the year using your function
        detected_year = get_exact_year(file_path)
        detected_year = get_exact_year(file_path)
        
        # Standardize slashes and split the path into pieces
        normalized_path = file_path.replace("\\", "/")
        path_parts = normalized_path.split("/")
        
        # In a path like ".../sec-edgar-filings/MSFT/10-K/.../full-submission.txt"
        # The ticker is always exactly 4 folders up from the end!
        detected_ticker = path_parts[-4] 
        
        detected_category = "finance" # Safe to hardcode for SEC downloads
        
        # 2. Correctly initialize the Ingestion class instance
        ingestor = Ingestion(
            docs=file_path, 
            doc_category=detected_category, 
            ticker=detected_ticker, 
            year=detected_year
        )
        
        # 3. Run your processing pipeline
        ingestor.partition()
        ingestor.chunkdocs()
        final_documents = ingestor.document()
        chroma_db = ingestor.embedding(summary=final_documents)
        print("[+] Ingestion Complete! Saved to database.")
        
    else:
        # If the text file isn't there or the database already exists, load it from memory
        
        print("\nLoading existing database...")
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"})
        chroma_db = Chroma(persist_directory="ddb/chroma_db", embedding_function=embedding_model)
        
        db_data = chroma_db.get() 
        final_documents = []
        for i in range(len(db_data['ids'])):
            doc = Document(
                page_content=db_data['documents'][i],
                metadata=db_data['metadatas'][i]
            )
            final_documents.append(doc)
        if len(final_documents) == 0:
            print("[!] Please delete the 'ddb' folder from your project sidebar and run the script again.")
            exit()
    # 4. Spin up your retriever and start chatting!
    search_engine = Retriever(vector_db=chroma_db, langchain_documents=final_documents)
    chat(search_engine)