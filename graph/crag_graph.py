import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

from grader.relevance_grader import grade_chunk
from grader.query_rewriter import rewrite_query
from retrieval.vector_store import GridKnowledgeStore
from retrieval.embedder import CorpusEmbedder
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever

class GraphState(TypedDict):
    question: str
    rewritten_query: str
    chunks: List[dict]
    good_chunks: List[dict]
    retry_count: int
    web_search_used: bool
    generation: str

# Initialize retrieval tools
store = GridKnowledgeStore()
embedder = CorpusEmbedder(skip_verification=True)
bm25 = BM25Index.load("data/embeddings/bm25_index.pkl")
retriever = HybridRetriever(store, embedder, bm25)

web_search_tool = DuckDuckGoSearchRun()

def retrieve_node(state: GraphState):
    query = state.get("rewritten_query") or state["question"]
    chunks = retriever.search(query, top_k=5)
    
    # Initialize count if first time
    retry_count = state.get("retry_count", 0)
    
    return {"chunks": chunks, "retry_count": retry_count, "good_chunks": []}

def grade_node(state: GraphState):
    question = state["question"]
    chunks = state["chunks"]
    
    good_chunks = []
    for chunk in chunks:
        res = grade_chunk(question, chunk["text"])
        if res.score == 1:
            good_chunks.append(chunk)
            
    return {"good_chunks": good_chunks}

def check_relevance(state: GraphState):
    if len(state["good_chunks"]) > 0:
        return "generate"
    elif state["retry_count"] == 0:
        return "rewrite"
    else:
        return "web_search"

def rewrite_node(state: GraphState):
    question = state["question"]
    new_query = rewrite_query(question)
    return {"rewritten_query": new_query, "retry_count": 1}

def search_fallback_node(state: GraphState):
    query = state.get("rewritten_query") or state["question"]
    try:
        search_result = web_search_tool.invoke(query)
    except Exception as e:
        search_result = f"Search failed: {e}"
        
    web_chunk = {"text": f"WEB SEARCH RESULT: {search_result}", "slug": "web"}
    good_chunks = state["good_chunks"] + [web_chunk]
    
    return {"good_chunks": good_chunks, "web_search_used": True}

def generate_node(state: GraphState):
    question = state["question"]
    chunks = state["good_chunks"]
    
    context = "\n\n".join([f"Source: {c.get('slug', 'unknown')}\n{c['text']}" for c in chunks])
    
    prompt = f"""You are an expert assistant for energy grid regulations. Given the following extracted parts of a long document and a question, create a final answer.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Question: {question}
=========
Context:
{context}
=========
Answer:"""
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    
    response = llm.invoke(prompt)
    return {"generation": response.content}

# Build Graph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("rewrite", rewrite_node)
workflow.add_node("web_search", search_fallback_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    check_relevance,
    {
        "generate": "generate",
        "rewrite": "rewrite",
        "web_search": "web_search"
    }
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

crag_app = workflow.compile()
