import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from graph.crag_graph import crag_app, retriever
from langchain_groq import ChatGroq

st.set_page_config(page_title="Corrective RAG: Energy Grid", layout="wide")

st.title("⚡ Corrective RAG: ENTSO-E Grid Regulations")
st.markdown("Compare **Naive RAG** vs **Corrective RAG (CRAG)** side-by-side.")

query = st.text_input("Ask a question about energy grid operations:", value="What is the billing procedure for reserve providers?")

if st.button("Run Comparison"):
    if not query:
        st.warning("Please enter a query.")
    else:
        col1, col2 = st.columns(2)
        
        # --- NAIVE RAG ---
        with col1:
            st.subheader("Naive RAG")
            with st.spinner("Running Naive RAG..."):
                # Retrieve directly without grading
                naive_chunks = retriever.search(query, top_k=5)
                context = "\n\n".join([f"Source: {c.get('slug', 'unknown')}\n{c['text']}" for c in naive_chunks])
                
                prompt = f"""You are an expert assistant. Given the following extracted parts of a long document and a question, create a final answer.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Question: {query}
=========
Context:
{context}
=========
Answer:"""
                llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
                naive_resp = llm.invoke(prompt)
                
                st.markdown("**Answer:**")
                st.write(naive_resp.content)
                
                with st.expander("Retrieved Context (No Grading)"):
                    for c in naive_chunks:
                        st.write(f"- {c['text'][:300]}...")
        
        # --- CORRECTIVE RAG ---
        with col2:
            st.subheader("Corrective RAG")
            with st.spinner("Running Corrective RAG..."):
                result = crag_app.invoke({"question": query})
                
                retry_count = result.get("retry_count", 0)
                web_used = result.get("web_search_used", False)
                final_chunks = result.get("good_chunks", [])
                
                # Show Correction flow
                if retry_count > 0 or web_used:
                    st.error("⚠️ **Initial Retrieval Failed (Low Confidence)**")
                    if retry_count > 0:
                        st.info(f"🔄 **Action: Rewrote Query** -> `{result['rewritten_query']}`")
                    if web_used:
                        st.warning("🌐 **Action: Fallback to Web Search**")
                else:
                    st.success("✅ **Initial Retrieval Passed Confidence Check**")
                
                st.markdown("**Answer:**")
                st.write(result["generation"])
                
                with st.expander("Retrieved Context (Quality Checked)"):
                    for c in final_chunks:
                        st.write(f"- {c['text'][:300]}...")
