"""
Complete evaluation pipeline for Corrective RAG.
Runs CRAG flow, collects metrics, and generates comprehensive report.
"""

import os
import sys
import time
import logging
from typing import List
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from graph.crag_graph import crag_app, retriever
from grader.relevance_grader import grade_chunk
from eval.metrics import MetricsComputer, QueryResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test queries (from original eval script)
TEST_QUERIES = [
    "What is the FCR activation threshold?",
    "How are imbalance settlement periods defined?",
    "What is the required capacity for FRR?",
    "Explain the billing procedures for reserve providers.",
    "What is the role of a balancing service provider (BSP)?",
    "Describe the cross-border activation process.",
    "What is the maximum allowed frequency deviation?",
    "Who is responsible for the LFC block imbalances?",
    "What happens during an emergency state grid failure?",
    "How does the settlement of unintended deviations work?"
]


class GenerateAnswer(BaseModel):
    """Simple LLM call for generation"""
    answer: str = Field(description="The answer to the question")


def naive_rag(query: str) -> tuple[List[dict], str]:
    """
    Naive RAG: retrieve chunks and generate without grading/correction.
    Returns (retrieved_chunks, generated_answer)
    """
    chunks = retriever.search(query, top_k=5)

    context = "\n\n".join(
        [f"Source: {c.get('slug', 'unknown')}\n{c['text']}" for c in chunks]
    )

    prompt = f"""You are an expert assistant for energy grid regulations. Given the following extracted parts of a long document and a question, create a final answer.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Question: {query}
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
    return chunks, response.content


def run_evaluation(queries: List[str] = None) -> None:
    """
    Run complete CRAG evaluation on test queries.
    """
    if queries is None:
        queries = TEST_QUERIES

    queries = queries[:3] if len(queries) > 3 else queries  # Limit to 3 for demo to save API calls

    computer = MetricsComputer()

    logger.info(f"Starting evaluation on {len(queries)} queries...")
    print(f"\n{'='*60}")
    print(f"CORRECTIVE RAG EVALUATION")
    print(f"{'='*60}\n")

    for i, query in enumerate(queries, 1):
        logger.info(f"\n[{i}/{len(queries)}] Processing: {query}")
        print(f"[{i}/{len(queries)}] Query: {query}")

        try:
            # ========== NAIVE RAG ==========
            print("  [*] Running Naive RAG...", end=" ", flush=True)
            naive_chunks, naive_gen = naive_rag(query)
            print("[OK]")

            # Grade naive chunks
            naive_scores = []
            for chunk in naive_chunks:
                try:
                    result = grade_chunk(query, chunk["text"])
                    score = float(result.score)  # Convert to 0.0-1.0
                    naive_scores.append(score)
                except Exception as e:
                    logger.warning(f"Failed to grade naive chunk: {e}")
                    naive_scores.append(0.5)

            avg_naive_score = sum(naive_scores) / len(naive_scores) if naive_scores else 0.0

            # ========== CORRECTIVE RAG ==========
            print("  [*] Running Corrective RAG...", end=" ", flush=True)
            crag_result = crag_app.invoke({"question": query})
            print("[OK]")

            crag_chunks = crag_result.get("good_chunks", [])
            crag_gen = crag_result.get("generation", "")
            retry_count = crag_result.get("retry_count", 0)
            web_used = crag_result.get("web_search_used", False)

            # Grade CRAG chunks
            crag_scores = []
            for chunk in crag_chunks:
                try:
                    result = grade_chunk(query, chunk["text"])
                    score = float(result.score)
                    crag_scores.append(score)
                except Exception as e:
                    logger.warning(f"Failed to grade CRAG chunk: {e}")
                    crag_scores.append(0.5)

            avg_crag_score = sum(crag_scores) / len(crag_scores) if crag_scores else 0.0

            # ========== SUMMARIZE ==========
            corrections_triggered = retry_count > 0 or web_used
            improvement_arrow = "UP" if avg_crag_score > avg_naive_score else "DN"

            print(f"  [*] Naive Chunks: {len(naive_chunks)} | Relevance: {avg_naive_score:.2f}")
            print(f"  [*] CRAG Chunks:  {len(crag_chunks)} | Relevance: {avg_crag_score:.2f} {improvement_arrow}")

            if corrections_triggered:
                correction_type = "Rewrite" if retry_count > 0 else ""
                fallback_type = "Web Search" if web_used else ""
                actions = " + ".join(filter(None, [correction_type, fallback_type]))
                print(f"  [*] Correction Triggered: {actions}")
            else:
                print(f"  [*] No Correction Needed [OK]")

            # ========== STORE RESULT ==========
            query_result = QueryResult(
                query=query,
                naive_chunks=naive_chunks,
                naive_generation=naive_gen,
                crag_chunks=crag_chunks,
                crag_generation=crag_gen,
                corrections_triggered=corrections_triggered,
                retry_count=retry_count,
                web_search_used=web_used,
                naive_relevance_scores=naive_scores,
                crag_relevance_scores=crag_scores,
                avg_naive_relevance=avg_naive_score,
                avg_crag_relevance=avg_crag_score,
            )

            computer.add_result(query_result)

            # Avoid rate limiting
            time.sleep(2)

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"  └─ ERROR: {e}")

    # ========== GENERATE REPORT ==========
    print(f"\n{'='*60}\n")

    if computer.results:
        metrics = computer.compute_overall_metrics()

        # Print full report
        computer.print_report(metrics)

        # Print table format
        print("\nMetrics in table format:")
        print(computer.export_table(metrics))

        # Save results to JSON
        computer.save_results("eval/eval_results.json")

        logger.info("Evaluation complete!")
    else:
        logger.error("No results collected!")


if __name__ == "__main__":
    run_evaluation()
