import os
import uuid
from dotenv import load_dotenv
load_dotenv()

import shutil
from fastapi import FastAPI,HTTPException,UploadFile,File,Form
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from contextlib import asynccontextmanager

from agent import graph
from fastapi.middleware.cors import CORSMiddleware
from ingestion import Ingestion
import agent as agent_module
from retriever import Retriever

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("Couldn't find the database")

POOL_URL = DB_URL.replace("+psycopg", "")
pool = AsyncConnectionPool(conninfo=POOL_URL, max_size=20,kwargs={"autocommit": True, "row_factory": dict_row}, open=False)
agent=None
@asynccontextmanager
async def lifespan(FastAPI):
    global agent
    await pool.open()
    memory = AsyncPostgresSaver(pool)
    await memory.setup()
    agent = graph.compile(checkpointer=memory, interrupt_before=["hitl"])
    # yield is like return but it won't end the function it freezes it then continues when it is called
    yield
    await pool.close()
app=FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aegis-ui-l596.onrender.com","http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Feedbackrequest(BaseModel):
    thread_id:str
    status:str
    feedback:Optional[str]=None

@app.post("/app/call")
async def restarting(question: str = Form(...), 
    thread_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)):
    # Form tells the FastApi to accept the data directly instead of expecting json format
    thread_id=thread_id or str(uuid.uuid4())
    config={"configurable":{"thread_id":thread_id}}
    if file:
        os.makedirs("Docs",exist_ok=True)
        file_path=os.path.join("Docs",file.filename)

        with open(file_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
        ingestor=Ingestion(
            docs=file_path
        )
        ingestor.partition()
        ingestor.chunkdocs()
        final_docs=ingestor.document()
        db=await ingestor.embedding(final_docs)
        await agent_module.vector_store.aadd_documents(final_docs)
        question = f"{question}\n\n[System: The user attached a file. It has been ingested into the Chroma Vector Database. Use your Retrieval tools to search it.]"

    initial_ques={"question":[HumanMessage(content=(question))]}
    memory=AsyncPostgresSaver(pool)
    state=await agent.ainvoke(initial_ques,config=config)
    # same as .get function of dictionary
    messages_display=state.get("output","Drafting the answer")
    raw_docs = state.get("documents", [])
    real_documents = []
    for doc in raw_docs:
        if hasattr(doc, "page_content"):
            real_documents.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        else:
            real_documents.append({"page_content": str(doc)})
    return{
        "status":"awaiting response",
        "thread_id":thread_id,
        "message_display":messages_display,
        "retrieved_context":real_documents
    }

@app.post("/app/feedback")
async def receivefeedback(feedback:Feedbackrequest):
    config = {"configurable": {"thread_id": feedback.thread_id}}
    try:
        if feedback.status=="Yes":
            agent.update_state(config, {"human_feedback": "Yes"}, as_node="hitl")
        elif feedback.status=="No":
            await agent.aupdate_state(
                config,
                {"human_feedback":"No",
                 # send the human feedback to the llm 
                "question": [HumanMessage(content=f"Human Feedback: {feedback.feedback}")]
                }
                ,as_node="hitl")
            
        final_state = await agent.ainvoke(None, config=config)
        return {
                "status": "success",
                "final_output": final_state.get("output", "No final output generated.")
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
