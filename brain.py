from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated,List,Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage,AIMessage
from pydantic import BaseModel,Field
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from retriever import Retriver
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=0)

retriver=Retriver

class MainGraph(TypedDict):
    question:Annotated[List[BaseMessage],add_messages]
    output:str

class Display(BaseModel):
    output:Annotated[str,Field(description="Display a strucured output for the model")]

structured_llm=llm.with_structured_output(Display)

class RAGSubGraph(TypedDict):
    question:Annotated[List[BaseMessage],add_messages]
    rewritten:bool
    retrieved:List[str] 
    draft:List[str]
    output:str
    grading:str
    structured_out:List[str]
    answer:List[str]

def output(state:MainGraph):
    output=state["output"]
    final=(structured_llm.invoke(output)).output
    return {"output":final}
 
parent=StateGraph(MainGraph)
child=StateGraph(RAGSubGraph)

child.add_node("retriever",retriever_graph)
child.add_node("Grading",grade)
child.add_node("Generate answer",gen_answer)
child.add_node("Answer check",answer_check)
child.add_node("Rewrite",rewrite_query)
child.add_node("output",to_parent)

child.add_edge(START,"retriever")
child.add_edge("retriever","Grading")
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

def retriever_graph(state:RAGSubGraph):
    """
The node calls the retriever function and retrives the necessary documents
    """
    query=state["question"]
    current_query=query[-1].content
    documents=retriver.search.invoke(current_query)

    return {"retrived":documents}

def retrieved_docs(BaseModel):
    docs:Annotated[str,Field(description="Format the docs grade it if revelant return pass else fail")]

def grade(state:RAGSubGraph):

    question=state["question"]
    current_question=question[-1].content
    docs=state["retrieved"]

    system_prompt = """You are Aegis, an elite financial auditor. 
    Your task is to evaluate retrieved SEC 10-K document chunks.
    Determine if the document contains facts, tables, or metrics relevant to the user's question.
    If it is relevant, grade it pass'. If it is completely irrelevant, grade it 'fail'."""

    human_prompt = """
    Here is the user's question:
    <user_question>
    {question}
    </user_question>

    Here is the retrieved document chunk:
    <retrieved_document>
    {docs}
    </retrieved_document>

    Carefully analyze the document against the question and provide your binary score.
    """
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
        # Create the strict grader
    structured_grader = llm.with_structured_output(retrieved_docs)
    grading_chain = grade_prompt | structured_grader
    for i in range(len(docs)):
        result=grading_chain.invoke({
            "question": current_question, 
            "chunks": docs,
        })
    state["structured_out"]=result
    if result.binary_score == "pass":
        return {"grading":"pass"}
    else:
        return {"grading":"fail"}

def examiner(state:RAGSubGraph)->Literal["pass","fail"]:
    if state["grading"]=="pass":
        return "Generate answer"
    else:
        return "Rewrite"
    
def gen_answer(state:RAGSubGraph):
    
    possible_ans=state["structured_out"]
    system_prompt="""You are Aegis, an elite financial auditor. 
    Your task is to generate answer to the user's question based on retrieved SEC 10-K document chunks.
    Don't add any extra information give the answer strictly on the basis of the retrieved chunks or if you can't find the required resource just print I don't know
    """

