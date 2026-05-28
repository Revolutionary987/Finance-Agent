from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated,List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from pydantic import BaseModel,Field
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)

class MainGraph(TypedDict):
    question:Annotated[List[BaseMessage],add_messages]
    output:str

class Display(BaseModel):
    output:Annotated[str,Field(description="Display a strucured output for the model")]

structured_llm=llm.with_structured_output(Display)

class RAGSubGraph(TypedDict):
    question:Annotated[List[BaseMessage],add_messages]
    rewritten:bool 
    draft:List[str]
    output:str

def output(state:MainGraph):
    output=state["output"]
    final=(structured_llm.invoke(output)).output
    return {"output":final}
 
parent=StateGraph(MainGraph)
child=StateGraph(RAGSubGraph)

child.add_node("retriever",retrirver)
child.add_node("reranker",reranker)
child.add_node("Grading",grade)
child.add_node("Generate answer",gen_answer)
child.add_node("Answer check",answer_check)
child.add_node("Rewrite",rewrite_query)
child.add_node("output",to_parent)

child.add_edge(START,"retriever")
child.add_edge("retriever","reranker")
child.add_edge("reranker","Grading")
child.add_conditional_edges("Grading",examiner)
child.add_conditional_edges("Generate answer",check_hallucination,{"hallucination":"Generate answer", "No hallucination": "Answer check"})
child.add_conditional_edges("Answer check",sufficient)
child.add_edge("Rewrite","retriever")
child.add_edge("output",END)

rag=child.compile()

parent.add_node("Subgraph",rag)
parent.add_node("Output",output)
parent.add_edge(START,"Subgraph")
parent.add_edge("Subgraph","Output")
parent.add_edge("Output",END)

