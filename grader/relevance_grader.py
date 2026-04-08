import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class GradeResult(BaseModel):
    score: int = Field(description="1 if relevant, 0 if irrelevant")
    reason: str = Field(description="One sentence reason for the score")

GRADER_PROMPT = """You are grading whether a retrieved document chunk
is relevant to answer a user question about energy grid operations.

Question: {question}
Chunk: {chunk}

Score 1 if the chunk contains information directly useful to answer
the question. Score 0 if it is irrelevant or off-topic.
Return ONLY a JSON: {{"score": 0 or 1, "reason": "one sentence"}}"""

def get_relevance_grader():
    # Initialize the Groq model
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    
    # Use structured output for reliable JSON parsing
    structured_llm = llm.with_structured_output(GradeResult, method="json_mode")
    
    prompt = PromptTemplate(
        template=GRADER_PROMPT,
        input_variables=["question", "chunk"]
    )
    
    # Create grader chain
    grader_chain = prompt | structured_llm
    return grader_chain

def grade_chunk(question: str, chunk: str) -> GradeResult:
    grader = get_relevance_grader()
    return grader.invoke({"question": question, "chunk": chunk})

if __name__ == "__main__":
    # Simple test
    test_q = "What is the FCR activation threshold?"
    test_chunk_yes = "The FCR activation threshold is typically set at 200 mHz."
    test_chunk_no = "Billing procedures must be completed by the 5th of each month."
    
    print("Test Yes:", grade_chunk(test_q, test_chunk_yes))
    print("Test No:", grade_chunk(test_q, test_chunk_no))
