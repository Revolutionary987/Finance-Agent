import os
import uuid
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from agent import brain

load_dotenv()
DB_URL=os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("Couldn't find the database")
pool=PostgresSaver(conninfo=DB_URL,max_size=20,open=False)

@asynccontextmanager
async def lifespan(FastAPI):
    pool.open()
    # yield is like return but it won't end the function it freezes it then continues when it is called
    yield
    pool.close()
app=FastAPI(lifespan=lifespan)
class Restart(BaseModel):
    question:str
    thread_id:Optional[str]=None
class Feedbackrequest(BaseModel):
    thread_id:str
    status:str
    feedback:Optional[str]=None

@app.post("/app/call")
async def restarting(restart:Restart):
    thread_id=restart.thread_id or str(uuid.uuid4())
    config={"configurable":{"thread_id":thread_id}}
    initial_ques={"question":[HumanMessage(content=(restart.question))]}
    memory=PostgresSaver(pool)
    state=brain.invoke(initial_ques,config=config)
    # same as .get function of dictionary
    messages_display=state.get("output","Drafting the answer")
    return{
        "status":"awaiting response",
        "thread_id":thread_id,
        "message_display":messages_display
    }

@app.post("/app/feedback")
async def receivefeedback(feedback:Feedbackrequest):
    config = {"configurable": {"thread_id": feedback.thread_id}}
    try:
        if feedback.status=="Yes":
            brain.update_state(config, {"human_feedback": "Yes"}, as_node="hitl")
        elif feedback.status=="No":
            brain.update_state(
                config,
                {"human_feedback":"No",
                 # send the human feedback to the llm 
                "question": [HumanMessage(content=f"Human Feedback: {feedback.feedback}")]
                }
                ,as_node="hitl")
            
        final_state = brain.invoke(None, config=config)
        return {
                "status": "success",
                "final_output": final_state.get("output", "No final output generated.")
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

