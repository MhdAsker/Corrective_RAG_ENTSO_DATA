import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class RewriteResult(BaseModel):
    rewritten_query: str = Field(description="The reformulated query")

REWRITE_PROMPT = """You are an expert at reformulating user questions for better semantic search retrieval.
The user asked a question about energy grid operations, but the initial search failed to find relevant documents.

Original Question: {question}

Reformulate this question to be more specific or to use better terminology related to ENTSO-E Network Codes, BDEW guidelines, or general power grid operations. 
Make sure it remains a question and maintains the original intent. Include relevant keywords if implicit.

Return ONLY a JSON: {{"rewritten_query": "your new query here"}}"""

def get_query_rewriter():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2, # slight variation for rewriting
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    
    structured_llm = llm.with_structured_output(RewriteResult, method="json_mode")
    
    prompt = PromptTemplate(
        template=REWRITE_PROMPT,
        input_variables=["question"]
    )
    
    return prompt | structured_llm

def rewrite_query(question: str) -> str:
    rewriter = get_query_rewriter()
    result = rewriter.invoke({"question": question})
    return result.rewritten_query

if __name__ == "__main__":
    # Simple test
    test_q = "What's the threshold for FCR?"
    print("Original:", test_q)
    print("Rewritten:", rewrite_query(test_q))
