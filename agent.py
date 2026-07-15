import os
from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()
import re
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
from langchain_openai import OpenAIEmbeddings

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
simple_llm=ChatOpenAI(model="gpt-4.1-nano",temperature=0)

raw_url= os.getenv("DATABASE_URL")
if not raw_url:
    raise ValueError("DATABASE_URL environment variable is missing!")
DB_URL = re.sub(r"^postgres(ql)?://", "postgresql+psycopg://", raw_url.strip())
embedding_model=OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ.get("OPENAI_API_KEY")
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
    original_question:str 
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
    original_q = state.get("original_question")
    if not original_q:
        original_q = current_query
    print(f"\n[RETRIEVER] Original question: '{original_q}'")
    print(f"\n[RETRIEVER] Searching with: '{current_query}'")
    search_results = await retriever.search(current_query,config=config)
    print(f"\n[DIAGNOSTIC] Retriever found {len(search_results['documents'])} documents in the database.")
    for i, doc in enumerate(search_results['documents']):
        print(f"  Doc {i}: {doc.page_content[:200]}")
    return {"retrieved": search_results["documents"],"original_question": original_q}

class DocumentGrade(BaseModel):
    doc_id: int = Field(description="The ID of the document chunk (e.g., 0, 1, 2)")
    binary_score: str = Field(description="Strictly 'pass' or 'fail'")

class BatchGrader(BaseModel):
    evaluations: List[DocumentGrade] = Field(description="List of evaluations for all provided documents")

structured_grader = simple_llm.with_structured_output(BatchGrader)

async def grade(state: RAGSubGraph):
    question = state["question"]
    current_question = question[-1].content
    
    # 1. Safely extract documents with an explicit fallback check
    docs = state.get("retrieved", [])
    if docs is None:
        docs = []
        
    # 2. Strict Guard Clause: If the list is empty, exit immediately
    if not docs or len(docs) == 0:
        print("0 documents found.")
        return {"structured_out": []}

    # 3. Build document representation strings safely
    docs_string = ""
    for i, doc in enumerate(docs):
        docs_string += f"\n<doc id='{i}'>\n{doc.page_content}\n</doc>\n"

    
    system_prompt = """You are an expert document relevance evaluator for a SEC 10-K financial filing RAG system.
 
    YOUR TASK:
    Evaluate each provided document chunk and decide whether it contains information that is 
    USEFUL to answering the user's question. You must evaluate EVERY document provided.
    
    GRADING FRAMEWORK — apply the correct rule based on question type:
 
    TYPE A — QUANTITATIVE questions (asking for specific numbers, dates, dollar amounts):
    PASS: Chunk contains the specific figure, date, or metric being asked about.
    FAIL: Chunk is on the right topic but contains no actual numbers or specific data points.
    
    TYPE B — QUALITATIVE / RISK / STRATEGY questions (asking HOW, WHY, WHAT risks, WHAT strategies):
    PASS: Chunk contains descriptive statements, risk factors, policies, or explanations 
            that are directly relevant to the subject of the question.
    FAIL: Chunk is about a completely different topic that shares no relevance to the question.
    
    TYPE C — MIXED questions (asking for both explanation and figures):
    PASS: Chunk contains EITHER relevant qualitative context OR relevant quantitative data.
    FAIL: Chunk is entirely off-topic.
    
    UNIVERSAL PASS CRITERIA — always pass if the chunk:
    - Directly mentions the subject entity (company, product, process) in the question AND
    - Provides any information (qualitative or quantitative) that helps understand 
    or answer what was asked.
 
    UNIVERSAL FAIL CRITERIA — always fail if the chunk:
    - Is boilerplate legal language (e.g. "This document is filed with the SEC...")
    - Is a table of contents, page header, or footer with no substantive content
    - Is about a completely unrelated business segment, product, or topic
    
    IMPORTANT: Do NOT require numbers to pass a qualitative question. 
    A chunk explaining supply chain risks IS relevant to a supply chain risk question
    even if it contains no dollar figures.
    
    Output one evaluation object per document. Do not skip any document.
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

    grading_chain = grade_prompt | structured_grader 
    
    filtered_docs = []
    
    try:
        result = await grading_chain.ainvoke({
            "question": current_question, 
            "docs_string": docs_string
        })
        
        # 4. Enforce strict index validation inside loop boundaries
        for evaluation in result.evaluations:
            try:
                doc_id = int(evaluation.doc_id)
                print(f"Grader evaluated doc {doc_id}: {evaluation.binary_score.upper()}")
                
                # Double check that the index physically exists within the current active list bounds
                if 0 <= doc_id < len(docs):
                    if evaluation.binary_score.lower().strip() == "pass":
                        filtered_docs.append(docs[doc_id])
                else:
                    print(f"[WARNING] LLM returned doc_id {doc_id}, but current list size is only {len(docs)}. Skipping out-of-bounds index.")
            except (ValueError, TypeError):
                print(f"[WARNING] LLM returned an invalid non-integer doc_id: {evaluation.doc_id}. Skipping parsing.")
                
    except Exception as e:
        print(f"[ERROR] Ingestion grading chain failed: {e}. Defaulting to empty context window.")
        return {"structured_out": []}
            
    return {"structured_out": filtered_docs}

async def examiner(state: RAGSubGraph) -> Literal["Rewrite","Generate answer"]:
    if len(state["structured_out"]) > 0:
        return "Generate answer"
    else:
        return "Rewrite"
    
async def gen_answer(state: RAGSubGraph):
    question = state["question"]
    original_question = state.get("original_question", question[-1].content)
    possible_ans = state["structured_out"]
    context_string = "\n\n---\n\n".join([doc.page_content for doc in possible_ans])
    attempts = state.get("generation_attempts", 0)
    
    system_prompt = """You are Aegis, an elite financial analyst and SEC filing expert.
    Your task is to answer the user's question using ONLY the retrieved SEC 10-K document chunks provided.
    
    STRICT ANSWER RULES:
    
    1. ANSWER TYPE — adapt your answer to the question type:
    - For QUANTITATIVE questions: quote exact figures, dates, and amounts verbatim from the chunks.
    - For QUALITATIVE/RISK questions: synthesize the relevant statements from the chunks into 
        a clear, structured explanation. Use bullet points if multiple risk factors are present.
    - For MIXED questions: provide both the explanation and the specific figures.
    
    2. GROUNDING RULES:
    - Every factual claim must be directly traceable to the retrieved chunks.
    - Do NOT infer, extrapolate, or use external knowledge.
    - Do NOT mention the fiscal year, company name, or document metadata unless it is 
        DIRECTLY relevant to answering the question.
    
    3. IF DATA IS MISSING:
    - If the chunks do not contain enough information to fully answer the question, state:
        "The retrieved document sections do not contain sufficient information to fully answer 
        this question. Here is what was found: [summarize what IS in the chunks]."
    - Never fabricate data or make up figures.
    
    4. FORMAT:
    - Be concise and precise. No preamble like "Based on the documents..." or "According to...".
    - Start directly with the answer.
    - Use bullet points for multi-part answers or lists of risk factors.
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
        "question": original_question,
    })
    
    return {"answer": result.content,"generation_attempts": attempts + 1}

class HallucinationGrading(BaseModel):
    """Binary score for hallucination check on generated answers."""
    reasoning: Annotated[str, Field(description="Brief explanation of why the answer is or is not hallucinated.")]
    hallucination: Annotated[str, Field(description="Strictly output 'Hallucination' or 'No Hallucination'.")]

structured_hallucination_checker = simple_llm.with_structured_output(HallucinationGrading)

async def hal_check(state: RAGSubGraph):
    answer = state["answer"]
    possible_ans = state["structured_out"]
    context_string = "\n\n---\n\n".join([doc.page_content for doc in possible_ans])
    system_prompt = """You are a strict factual grounding auditor for a financial RAG system.
 
    YOUR TASK:
    Determine whether the generated answer is fully grounded in the provided source documents.
    
    GROUNDING RULES — apply based on answer type:
    
    FOR QUANTITATIVE ANSWERS:
    - Every number, dollar figure, date, and percentage must appear verbatim in the source documents.
    - If the answer states a figure not present in the documents: HALLUCINATION.
    
    FOR QUALITATIVE ANSWERS (risk factors, strategies, descriptions):
    - Every claim or statement must be inferable from the source documents.
    - Paraphrasing and summarizing source content is acceptable and is NOT a hallucination.
    - Only flag as hallucination if the answer introduces facts, entities, or claims 
        that have no basis in any of the source documents.
    
    FOR "DATA NOT FOUND" ANSWERS:
    - If the answer says the information was not found in the documents, grade it 'No Hallucination'
        since it makes no unsupported claims.
    
    OUTPUT: Return only valid JSON with 'reasoning' and 'hallucination' fields.
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
    
    gen_cycle = generation | structured_hallucination_checker
    
    result = await gen_cycle.ainvoke({
        "documents": context_string,
        "answer": answer
    })
    
    return {"hallucination": result.hallucination}

async def check_hallucination(state: RAGSubGraph) -> Literal["Generate answer", "Answer check","Rewrite"]:
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
structured_answer_validator = simple_llm.with_structured_output(cond_answer)
async def answer_check(state: RAGSubGraph):
    answer = state["answer"]
    question = state["question"]
    original_question = state.get("original_question", question[-1].content)

    system_prompt = """You are a senior quality assurance auditor for a financial RAG system.
 
    YOUR TASK:
    Evaluate whether the generated answer adequately addresses the user's original question.
    
    SUFFICIENCY FRAMEWORK — adapt to question type:
    
    FOR QUANTITATIVE QUESTIONS (specific numbers, dates, dollar amounts):
    SUFFICIENT: Answer contains the specific figure or data point requested.
    NOT SUFFICIENT: Answer is vague, deflects, or gives a generic description instead of the number.
    
    FOR QUALITATIVE QUESTIONS (how, why, what risks, what strategies, explanations):
    SUFFICIENT: Answer provides a clear, relevant explanation that addresses the subject 
                of the question with specific details from the documents.
    NOT SUFFICIENT: Answer is a single generic sentence, entirely off-topic, or 
                    just says "the document discusses X" without actual content.
    
    FOR "DATA NOT FOUND" ANSWERS:
    SUFFICIENT: If the answer honestly states the information was not found AND 
                explains what related information WAS found. This is a valid terminal state.
    NOT SUFFICIENT: If the answer simply says "not found" with zero supporting context.
    
    ALWAYS MARK NOT SUFFICIENT IF:
    - The answer talks about document metadata (form type, filing year) instead of the question topic.
    - The answer is a deflection like "the document is a 10-K filing."
    - The answer introduces information unrelated to what was asked.
    
    ALWAYS MARK SUFFICIENT IF:
    - For qualitative questions: the answer provides substantive, relevant content 
    even without specific numbers.
    - The answer directly and completely addresses what was asked.
 
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
    Is this answer sufficient? Return JSON with 'scoring' field set to 'sufficient' or 'not sufficient'
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt)
    ])
    
    # structured_out = simple_llm.with_structured_output(cond_answer)
    gen_out = prompt | structured_answer_validator
    
    result = await gen_out.ainvoke({
        "answer": answer,
        "question": original_question
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
    print(f"\n[DIAGNOSTIC] REWRITE: Triggered! (Count: {state.get('rewritten', 0)})")
    if rewritten_count >= 4:
        failure_msg = "Aegis audited the SEC filings multiple times but could not find the specific financial data required to answer this query."
        return {
                "answer": failure_msg, 
                "rewritten": rewritten_count + 1
        }
    original_user_question = state.get("original_question", question[0].content)
    history_logs = []
    if len(question) > 1:
        for msg in question[1:]:
            prefix = "Feedback Loop Log" if "Human Feedback" in msg.content else "Previous Rewrite Attempt"
            history_logs.append(f"- {prefix}: '{msg.content}'")
    chat_history_string = "\n".join(history_logs) if history_logs else "No previous execution context."
    system_prompt = """You are Aegis, an elite SEC financial document search specialist.
 
    YOUR TASK:
    Rewrite the user's question into a new search query optimized for semantic vector search 
    over SEC 10-K filings stored in a PostgreSQL vector database.
    
    REWRITE STRATEGY — adapt based on question type:
    
    FOR QUANTITATIVE QUESTIONS (specific numbers, dates, amounts):
    - Include the specific metric name exactly as it would appear in the filing.
    - Include any dates or fiscal periods mentioned.
    - Use SEC filing terminology: "aggregate market value", "diluted EPS", "total revenue", etc.
    - Example: "aggregate market value voting non-voting common stock non-affiliates March 2021"
    
    FOR QUALITATIVE / RISK QUESTIONS (how, why, risks, strategies):
    - Extract the core subject entities (e.g., "outsourcing partners", "supply chain", "Asia")
    - Focus on the TOPIC not the question structure.
    - Use terms that would appear in a Risk Factors or MD&A section.
    - Example: "supply chain concentration risk single-source manufacturing outsourcing Asia"
    
    FOR STRATEGY / OPERATIONS QUESTIONS:
    - Use operational terminology from 10-K filings.
    - Focus on business units, segments, or processes mentioned.
    
    ANTI-PATTERNS — never do these:
    - Do NOT add "GAAP", "fiscal year requirements", "accounting standards" unless specifically asked.
        These terms dilute the semantic signal and confuse the retriever.
    - Do NOT make the query longer and more complex each attempt — try SHORTER, more targeted queries.
    - Do NOT repeat a query that already failed. Check the history and change the approach.
    
    OUTPUT: Return a single optimized search query as a natural language sentence or phrase.
    """
    
    human_prompt = """
    Previous Conversation History:
    {chat_history}
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
            "chat_history": chat_history_string,
            "question": original_user_question
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
    return {
        "output":              final_ans,
        "documents":           docs,
        "original_question":   "",   # cleared for next turn
        "rewritten":           0,    # reset counter
        "generation_attempts": 0,    # reset counter
        "answer":              "",   # clear so stale answer doesn't persist
        "hallucination":       "",
        "is_sufficient":       "",
        "structured_out":      [],
        "retrieved":           [],
    }
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