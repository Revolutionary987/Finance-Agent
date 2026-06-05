import os
from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()

from langchain_postgres import PGVector
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from retriever import Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import uuid
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableConfig

session_thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": session_thread_id}}
#LLMS
primary_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0,api_key=os.getenv("GROQ_API_KEY"))
reserve_primary=ChatOpenAI(model="llama-3.3-70b", api_key=os.getenv("CEREBRAS_API_KEY"), base_url="https://api.cerebras.ai/v1",temperature=0)
# simple_reserve1=ChatOpenAI(base_url="https://api.sambanova.ai/v1",api_key=os.getenv("SAMBANOVA_API_KEY"),model="gemma-3-12b-it")
# simple_reserve2= ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# simple_task_llm = simple_reserve2.with_fallbacks([simple_reserve1])
primary_llm = primary_llm.with_fallbacks([reserve_primary])
beast = ChatOpenAI(base_url="https://api.sambanova.ai/v1",api_key=os.getenv("SAMBANOVA_API_KEY"),model="Meta-Llama-3.3-70B-Instruct", temperature=0)
slm_endpoint = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct",task="text-generation",max_new_tokens=150,huggingfacehub_api_token=os.getenv("HF_TOKEN"))
simple_llm = ChatHuggingFace(llm=slm_endpoint)

DB_URL = os.getenv("DATABASE_URL")
model_name="BAAI/bge-m3"
model_kwargs={"device":"cpu"}
encode_kwargs={"normalize_embeddings":True}
embedding_model=HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
)

vector_store = PGVector(
    embeddings=embedding_model,
    collection_name="aegis_db",
    connection=DB_URL,
    async_mode=True
)
retriever = Retriever(vector_db=vector_store, langchain_documents=[])
class MainGraph(TypedDict):
    question: Annotated[List[BaseMessage], add_messages]
    human_feedback:str
    output: str
    documents: List[Document]

class Display(BaseModel):
    output: Annotated[str, Field(description="Display a structured output for the model")]

structured_llm = primary_llm.with_structured_output(Display)

class RAGSubGraph(TypedDict):
    question: Annotated[List[BaseMessage], add_messages]
    retrieved: List[str] 
    draft: List[str]
    output: str
    grading: str
    structured_out: List[str]
    answer: str
    hallucination: str
    is_sufficient: str
    final_output: str
    rewritten:int 
    generation_attempts: int

async def output(state: MainGraph):
    output_text = state["output"]
    final = (await structured_llm.ainvoke(output_text)).output
    return {"output": final}

async def retriever_graph(state: RAGSubGraph):
    """The node calls the retriever function and retrieves the necessary documents"""
    query = state["question"]
    current_query = query[-1].content
    search_results = await retriever.search(current_query,config=config)
    print(f"\n[DIAGNOSTIC] Retriever found {len(search_results['documents'])} documents in the database.")
    
    return {"retrieved": search_results["documents"]}

class DocumentGrade(BaseModel):
    doc_id: int = Field(description="The ID of the document chunk (e.g., 0, 1, 2)")
    binary_score: str = Field(description="Strictly 'pass' or 'fail'")

class BatchGrader(BaseModel):
    evaluations: List[DocumentGrade] = Field(description="List of evaluations for all provided documents")

async def grade(state: RAGSubGraph):
    question = state["question"]
    current_question = question[-1].content
    docs = state["retrieved"]
    docs_string = ""
    for i, doc in enumerate(docs):
        docs_string += f"\n<doc id='{i}'>\n{doc.page_content}\n</doc>\n"

    docs_parser = PydanticOutputParser(pydantic_object=BatchGrader)
    
    system_prompt = """You are a strict relevance filter. 
    You will be given a list of documents. Evaluate EACH document individually to see if it contains ANY numbers, metrics, or keywords that could help answer the user's question.
    
    RULES:
    - If the document contains relevant financial data, grade it 'pass'.
    - If it is completely off-topic, grade it 'fail'.
    
    CRITICAL INSTRUCTION: You MUST output strictly valid JSON matching the format instructions. Evaluate every document provided.
    
    {format_instructions}
    """

    human_prompt = """
    User Question:
    <user_question>
    {question}
    </user_question>

    Documents to Evaluate:
    {docs_string}
    """

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])

    # structured_grader = simple_task_llm.with_structured_output(retrieved_docs)
    grading_chain = grade_prompt | simple_llm | docs_parser
    
    filtered_docs = []
    result = await grading_chain.ainvoke({
            "question": current_question, 
            "docs_string": docs_string,
            "format_instructions": docs_parser.get_format_instructions()
        })
    # result.evaluations is a list that contains document id and binary_score of each chunk or document
    for evaluation in result.evaluations:
        print(f"[DIAGNOSTIC] Grader evaluated doc {evaluation.doc_id}: {evaluation.binary_score.upper()}")
        if evaluation.binary_score.lower().strip() == "pass":
            filtered_docs.append(docs[evaluation.doc_id])
    return {"structured_out": filtered_docs}

async def examiner(state: RAGSubGraph) -> Literal["Rewrite","Generate answer"]:
    if len(state["structured_out"]) > 0:
        print("[DIAGNOSTIC] ---> GENERATE ANSWER")
        return "Generate answer"
    else:
        print("[DIAGNOSTIC] ---> REWRITE")
        return "Rewrite"
    
async def gen_answer(state: RAGSubGraph):
    question = state["question"]
    current_question = question[-1].content 
    possible_ans = state["structured_out"]
    context_string = "\n\n---\n\n".join([doc.page_content for doc in possible_ans])
    attempts = state.get("generation_attempts", 0)
    
    system_prompt = """You are Aegis, an elite financial auditor. 
    Your task is to generate answer to the user's question based on retrieved SEC 10-K document chunks.
    Don't add any extra information give the answer strictly on the basis of the retrieved chunks or if you can't find the required resource just print I couldn't find the solution
    """
    
    human_prompt = """
    Carefully analyze the retrieved document chunks below:
    <retrieved_documents>
    {possible_ans}
    </retrieved_documents>
    
    Based ONLY on those documents, answer the user's question:
    <user_question>
    {question}
    </user_question>
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    generating_ans = prompt | primary_llm
    
    result = await generating_ans.ainvoke({
        "possible_ans": context_string,
        "question": current_question,
    })
    
    return {"answer": result.content,"generation_attempts": attempts + 1}

class HallucinationGrading(BaseModel):
    """Binary score for hallucination check on generated answers."""
    reasoning: Annotated[str, Field(description="Brief explanation of why the answer is or is not hallucinated.")]
    hallucination: Annotated[str, Field(description="Strictly output 'Hallucination' or 'No Hallucination'.")]

async def hal_check(state: RAGSubGraph):
    answer = state["answer"]
    possible_ans = state["structured_out"]
    context_string = "\n\n---\n\n".join([doc.page_content for doc in possible_ans])
    hal_parser = PydanticOutputParser(pydantic_object=HallucinationGrading)

    system_prompt = """You are a strict auditor evaluating an AI-generated report. 
    Your only task is to determine whether the generated answer is entirely grounded in the provided source documents.
    If the answer contains ANY numbers, facts, or claims that are not explicitly stated in the source documents, it is a hallucination.
    If it is a hallucination, grade it 'Hallucination'. If it is perfectly grounded, grade it 'No Hallucination'.
    
    CRITICAL INSTRUCTION: You are a backend data processor. You MUST output strictly and ONLY valid JSON. 
    Do NOT output any conversational text, preamble, or markdown blocks.
    
    {format_instructions}
    
    EXAMPLE OUTPUT:
    {{"reasoning": "The revenue numbers match the text.", "hallucination": "No Hallucination"}}
    """

    human_prompt = """
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
    
    generation = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    gen_cycle = generation | simple_llm | hal_parser
    
    result = await gen_cycle.ainvoke({
        "documents": context_string,
        "answer": answer,
        "format_instructions": hal_parser.get_format_instructions()
    })
    
    return {"hallucination": result.hallucination}

async def check_hallucination(state: RAGSubGraph) -> Literal["Generate answer", "Answer check"]:
    if state["hallucination"] == "Hallucination":
        print(f"\n[DIAGNOSTIC] HAL_CHECK: Grade was '{state['hallucination']}' (Attempt {state.get('generation_attempts', 0)})")
        if state.get("generation_attempts", 0) >= 3:
            return "Rewrite"
            
        return "Generate answer"
    else:
        return "Answer check"

class cond_answer(BaseModel):
    "Binary score for the generated answer"
    scoring: Annotated[str, Field(description="Return strictly 'sufficient' if the generated answer correctly answers the question else 'not sufficient'")]

async def answer_check(state: RAGSubGraph):
    answer = state["answer"]
    question = state["question"]
    current_question = question[-1].content
    ans_parser = PydanticOutputParser(pydantic_object=cond_answer)

    system_prompt = """
    You are Aegis, an expert finance auditor
    You task is to evaluate the generated answer strictly based on the question 
    return 'sufficient' if it exactly answers the question else 'not sufficient'
    
    CRITICAL INSTRUCTION: You are a backend data processor. You MUST output strictly and ONLY valid JSON. 
    Do NOT output any conversational text, preamble, or markdown blocks.
    
    {format_instructions}
    
    EXAMPLE OUTPUT:
    {{"scoring": "sufficient"}}

    """
    
    human_prompt = """
    Evaluate the generated answers
    <generated_answer>
    {answer}
    </generated_answer>
    based on the user's query
    <original_question>
    {question}
    </original_question>
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    # structured_out = simple_llm.with_structured_output(cond_answer)
    gen_out = prompt | simple_llm | ans_parser
    
    result = await gen_out.ainvoke({
        "answer": answer,
        "question": current_question,
        "format_instructions": ans_parser.get_format_instructions()
    })
    return {"is_sufficient": result.scoring}

async def sufficient(state: RAGSubGraph) -> Literal["output", "Rewrite"]:
    if state["is_sufficient"] == "sufficient":
        return "output"
    else:
        return "Rewrite"
    
class RewrittenQuery(BaseModel):
    """The optimized query for vector search."""
    new_query: Annotated[str, Field(description="The rewritten, highly optimized search query.")]

async def rewrite_query(state: RAGSubGraph):
    question = state["question"]
    rewritten_count=state.get("rewritten",0)
    current_org_ques = question[-(rewritten_count + 1)].content
    print(f"\n[DIAGNOSTIC] REWRITE: Triggered! (Count: {state.get('rewritten', 0)})")
    if rewritten_count >= 4:
        failure_msg = "Aegis audited the SEC filings multiple times but could not find the specific financial data required to answer this query."
        return {
                "answer": failure_msg, 
                "rewritten": rewritten_count + 1
        }
    system_prompt = """You are Aegis, an elite financial researcher and query optimization expert.
    Your task is to take a user's question and rewrite it to be highly optimized for semantic vector search across SEC 10-K filings.
    
    CRITICAL RULES:
    1. DO NOT output a disconnected list of keywords. 
    2. DO output a complete, natural-sounding, grammatically correct sentence or question.
    3. Seamlessly weave relevant financial terms (like 'revenue', 'GAAP', 'fiscal year') into the natural sentence.
    
    BAD OUTPUT: "Q1 2026 revenue results GAAP financial performance"
    GOOD OUTPUT: "What were the reported revenue results and GAAP financial performance for Q1 2026?"
    
    Return ONLY the optimized natural language query.
    """
    
    human_prompt = """
    The previous search failed to find relevant financial documents. We need a better query.

    Here is the user's original question:
    <original_question>
    {question}
    </original_question>

    Rewrite this question into a focused, keyword-rich search query optimized for a financial document database.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    structured_out = primary_llm.with_structured_output(RewrittenQuery)
    gen_ans = prompt | structured_out
    result = await gen_ans.ainvoke({
            "question": current_org_ques
        })
    print(f"[DIAGNOSTIC] New Search Query: {result.new_query}")
    return {"question": [HumanMessage(content=result.new_query)],"rewritten":rewritten_count+1}

async def decider(state:RAGSubGraph)->Literal["retriever", "output"]:
    if state.get("rewritten",0)<=4:
        return "retriever"
    else:
        return "output"
    
async def to_parent(state: RAGSubGraph):
    final_ans = state.get("answer","Sorry couldn't find the answer for your query")
    docs = state.get("structured_out", [])
    return {"output": final_ans, "documents": docs}

async def hitl(state:MainGraph):
    return state

async def check_hitl(state:MainGraph)->Literal["Output","Subgraph","hitl"]:
    feedback = state.get("human_feedback")
    if feedback =="Yes":
        return "Output"
    elif feedback=="No":
        return "Subgraph"
    else:
        return "hitl"


child = StateGraph(RAGSubGraph)

child.add_node("retriever", retriever_graph)
child.add_node("Grading", grade)
child.add_node("Generate answer", gen_answer)
child.add_node("hallucination", hal_check)
child.add_node("Answer check", answer_check)
child.add_node("Rewrite", rewrite_query)
child.add_node("output", to_parent)


child.add_edge(START, "retriever")
child.add_edge("retriever", "Grading")
child.add_conditional_edges("Grading", examiner)
child.add_edge("Generate answer", "hallucination")
child.add_conditional_edges("hallucination", check_hallucination)
child.add_conditional_edges("Answer check", sufficient)
child.add_conditional_edges("Rewrite", decider)
child.add_edge("output",END)


rag = child.compile()

parent = StateGraph(MainGraph)
parent.add_node("hitl",hitl)
parent.add_node("Subgraph", rag)
parent.add_node("Output", output)
parent.add_edge(START, "Subgraph")
parent.add_edge("Subgraph", "hitl")
parent.add_conditional_edges("hitl",check_hitl)
parent.add_edge("Output", END)

graph = parent