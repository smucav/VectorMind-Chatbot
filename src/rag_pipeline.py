import os
import pandas as pd
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import gc
from google.colab import drive
from huggingface_hub import InferenceClient

# Mount Google Drive
drive.mount('/content/drive')

# Set paths consistent with Task 2
VECTOR_STORE_DIR = Path("/content/drive/MyDrive/vector_store")
vector_store_path = VECTOR_STORE_DIR / "chroma_db"

# Initialize embedding model consistent with Task 2
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
print(f"Embedding model initialized on cuda")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=str(vector_store_path))
collection = client.get_collection("cfpb_complaints")
print(f"Loaded ChromaDB collection: cfpb_complaints")

# Initialize Mistral-7B via Hugging Face Inference API
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_api_token_here"  # Replace with your token

HF_TOKEN = "your_hf_api_token_here"  # Store this securely

llm_client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)
print("Initialized Mistral-7B via Hugging Face Inference API")

# Define prompt template
prompt_template = """
You are a financial analyst assistant for CrediTrust, specializing in customer complaint analysis. Your task is to answer questions based solely on the provided complaint excerpts. If the context doesn't contain enough information, state that clearly and avoid speculation. Provide a concise, accurate, and professional response.

Context:
{context}

Question: {question}

Answer:
"""

# RAG pipeline function
def run_rag_pipeline(question, k=5):
    """
    Run the RAG pipeline: retrieve relevant chunks and generate an answer.
    
    Args:
        question (str): User question
        k (int): Number of chunks to retrieve (default: 5)

    Returns:
        dict: Contains answer, retrieved documents, and metadata
    """
    # Embed the question
    query_embedding = model.encode([question], device='cuda')[0].tolist()

    # Retrieve top-k chunks
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    # Combine retrieved chunks into context
    context = "\n".join(results['documents'][0])

    # Format prompt
    prompt = prompt_template.format(context=context, question=question)

    # Generate answer
    response = llm_client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    answer = response.choices[0].message["content"]

    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "question": question,
        "answer": answer.strip(),
        "retrieved_documents": [
            {"text": doc[:300], "metadata": meta}
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
    }

# Evaluation function
def evaluate_rag_pipeline():
    """
    Evaluate the RAG pipeline on representative questions.
    
    Returns:
        list: Evaluation results for table
    """
    questions = [
        "What are common issues with Buy Now, Pay Later (BNPL) services?",
        "Why do customers complain about credit card fees?",
        "What problems do people face with personal loan repayments?",
        "Are there frequent issues with savings account access?",
        "What are typical complaints about money transfer delays?"
    ]
    
    evaluation_results = []
    
    for question in questions:
        result = run_rag_pipeline(question, k=5)
        
        # Placeholder quality score and analysis
        quality_score = 3  # Adjust after reviewing output
        analysis = "Pending manual review of answer relevance and retrieved context accuracy."
        
        # Check relevance of retrieved documents
        relevant_docs = any(
            question.lower().split()[2] in meta.get('product_category', '').lower()
            for meta in result["retrieved_documents"]
        )
        if relevant_docs:
            quality_score = min(quality_score + 1, 5)
            analysis = "Retrieved documents align with product category; verify answer specificity."
        
        evaluation_results.append({
            "question": question,
            "answer": result["answer"],
            "retrieved_sources": [
                f"Text: {doc['text']}... Metadata: {doc['metadata']}"
                for doc in result["retrieved_documents"][:2]
            ],
            "quality_score": quality_score,
            "analysis": analysis
        })

    # Save evaluation table
    with open(VECTOR_STORE_DIR / "task3_evaluation.md", "w") as f:
        f.write("# Task 3: RAG Pipeline Evaluation\n\n")
        f.write("| Question | Generated Answer | Retrieved Sources | Quality Score | Comments/Analysis |\n")
        f.write("|----------|------------------|------------------|---------------|------------------|\n")
        for res in evaluation_results:
            sources = "<br>".join(res["retrieved_sources"])
            f.write(f"| {res['question']} | {res['answer'][:200]}... | {sources} | {res['quality_score']} | {res['analysis']} |\n")

    return evaluation_results

# Run evaluation
if __name__ == "__main__":
    results = evaluate_rag_pipeline()
    print("Evaluation completed and saved to vector_store/task3_evaluation.md")

    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
