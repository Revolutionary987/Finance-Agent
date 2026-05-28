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
    answer:str
    hallucination:str
    is_sufficient:str

def output(state:MainGraph):
    output=state["output"]
    final=(structured_llm.invoke(output)).output
    return {"output":final}
 
parent=StateGraph(MainGraph)
child=StateGraph(RAGSubGraph)

child.add_node("retriever",retriever_graph)
child.add_node("Grading",grade)
child.add_node("Generate answer",gen_answer)
child.add_node("hallucination",hal_check)
child.add_node("Answer check",answer_check)
child.add_node("Rewrite",rewrite_query)
child.add_node("output",to_parent)

child.add_edge(START,"retriever")
child.add_edge("retriever","Grading")
child.add_conditional_edges("Grading",examiner)
child.add_edge("Generate answer","hallucination")
child.add_edge("hallucination",check_hallucination)
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

    structured_grader = llm.with_structured_output(retrieved_docs)
    grading_chain = grade_prompt | structured_grader
    filtered_docs=[]
    for i in range(len(docs)):
        result = grading_chain.invoke({
            "question": current_question, 
            "docs": docs[i], 
        })
        if result.binary_score == "pass":
            filtered_docs.append(docs[i])
    return {"structured_out": filtered_docs}

def examiner(state:RAGSubGraph)->Literal["pass","fail"]:
    if len(state["structured_out"])>0:
        return "Generate answer"
    else:
        return "Rewrite"
    
def gen_answer(state:RAGSubGraph):
    question=state["question"]
    possible_ans=state["structured_out"]
    context_string = "\n\n---\n\n".join(possible_ans)
    system_prompt="""You are Aegis, an elite financial auditor. 
    Your task is to generate answer to the user's question based on retrieved SEC 10-K document chunks.
    Don't add any extra information give the answer strictly on the basis of the retrieved chunks or if you can't find the required resource just print I couldn't find the solution
    """
    human_prompt="""
    Carefully analyze the retrieved document chunks below:
    <retrieved_documents>
    {possible_ans}
    </retrieved_documents>
    
    Based ONLY on those documents, answer the user's question:
    <user_question>
    {question}
    </user_question>
    
"""
    prompt=ChatPromptTemplate.from_messages(
        ("system",system_prompt)
        ("human",human_prompt)
    )
    generating_ans=prompt|llm
    
    result=generating_ans.invoke(
        "possible_ans":context_string,
        "question":question
    )
    return{"answer":result.content}

class HallucinationGrading(BaseModel):
    """Binary score for hallucination check on generated answers."""
    reasoning: Annotated[str,Field(description="Brief explanation of why the answer is or is not hallucinated.")]
    hallucination: Annotated[str,Field(description="Strictly 'yes' if the answer contains hallucinated facts, or 'no' if it is completely grounded.")]

def hal_check(state:RAGSubGraph):
    answer=state["answer"]
    possible_ans=state["structured_out"]
    context_string = "\n\n---\n\n".join(possible_ans)
    system_prompt = """You are a strict auditor evaluating an AI-generated report. 
Your only task is to determine whether the generated answer is entirely grounded in the provided source documents.
If the answer contains ANY numbers, facts, or claims that are not explicitly stated in the source documents, it is a hallucination.
If it is a hallucination, grade it 'yes'. If it is perfectly grounded, grade it 'no'."""

    human_prompt="""
Here are the source documents retrieved from the SEC 10-K:
<documents>
{documents}
</documents>

Here is the generated answer to evaluate:
<generation>
{answer}
</generation>

Carefully analyze the generation against the documents. Does the generation contain any information, metrics, or claims that cannot be proven by the source documents? Provide your reasoning and your binary score.
"""
    generation=ChatPromptTemplate.from_messages(
        ("system",system_prompt),
        ("human",human_prompt)
    )
    structured_output=llm.with_structured_output(HallucinationGrading)
    gen_cycle=generation|structured_output
    result=gen_cycle.invoke(
        "documents":context_string,
        "answer":answer,
    )
    return {"hallucination":result}

def check_hallucination(state:RAGSubGraph)->Literal["Hallucination","No Hallucination"]:
    if state["hallucination"]=="Hallucination":
        return "Generate answer"
    else:
        return "Answer check"

def cond_answer(BaseModel):
    "Binary score for the generated answer"
    scoring:Annotated[str,Field("Return strictly sufficient if the generated answercorrectly answers the question else not sufficient")]

def answer_check(state:RAGSubGraph):
    answer=state["answer"]
    question=state["question"]
    system_prompt="""
You are Aegis, an expert finance auditor
You task is it evaluate the generated answer strictly based on the question 
return sufficient if it exactly answers the question else not sufficient
"""
    human_prompt="""
Evaluate the generated answers
<answer>
{answer}
<answer>
based on the user's query
<question>
{question}
<question>
"""
    prompt=ChatPromptTemplate.from_messages(
        ("system",system_prompt),
        ("human",human_prompt),

    )
    structured_out=llm.invoke(cond_answer)
    gen_out=prompt|structured_out
    result=gen_out.invoke(
        "answer",answer,
        "question",question,
    )
    return {"is_sufficient",result}
def sufficient(state:RAGSubGraph)->Literal["sufficient","not sufficient"]:
    if state["is_sufficient"]=="sufficient":
        return "output"
    else:
        return "Rewrite"
def rewrite_query(state:RAGSubGraph):
    
