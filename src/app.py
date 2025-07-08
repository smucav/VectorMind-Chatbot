import os
import gradio as gr
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import gc
from huggingface_hub import InferenceClient
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set paths consistent with Task 2
VECTOR_STORE_DIR = Path("/content/drive/MyDrive/vector_store")
vector_store_path = VECTOR_STORE_DIR / "chroma_db"

# Initialize embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
print(f"Embedding model initialized on cuda")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=str(vector_store_path))
collection = client.get_collection("cfpb_complaints")
print(f"Loaded ChromaDB collection: cfpb_complaints")

# Initialize Mistral-7B via Hugging Face Inference API
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_hf_api_token_here"  # Replace with your token
llm_client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.2")
print("Initialized Mistral-7B via Hugging Face Inference API")

# Define prompt template
prompt_template = """
You are a financial analyst assistant for CrediTrust, specializing in customer complaint analysis. Your task is to answer questions based solely on the provided complaint excerpts. If the context doesn't contain enough information, state that clearly and avoid speculation. Provide a concise, accurate, and professional response.

Context:
{context}

Question: {question}

Answer:
"""

# RAG pipeline function (from Task 3)
def run_rag_pipeline(question, k=5):
    if not question.strip():
        return "Please enter a valid question.", []
    
    # Embed the question
    query_embedding = model.encode([question], device='cuda')[0].tolist()
    
    # Retrieve top-k chunks
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    
    # Combine retrieved chunks into context
    context = "\n".join(results['documents'][0])
    
    # Format prompt
    prompt = prompt_template.format(context=context, question=question)
    
    # Generate answer
    answer = llm_client.text_generation(
        prompt,
        max_new_tokens=500,
        temperature=0.7,
        return_full_text=False
    )
    
    # Format sources for display
    sources = [
        f"Source {i+1}: {doc[:300]}...\nMetadata: {meta}"
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]))
    ]
    
    # Clean up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return answer.strip(), sources

# Gradio interface
def chat_interface(question):
    answer, sources = run_rag_pipeline(question)
    sources_text = "\n\n".join(sources) if sources else "No sources retrieved."
    return answer, sources_text

# Clear button function
def clear_conversation():
    return "", "", ""

# Build Gradio app
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# CrediTrust Complaint Analysis Chatbot")
    gr.Markdown("Ask questions about customer complaints (e.g., 'What are common issues with BNPL?').")
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="Type your question here...",
            lines=2
        )
    
    with gr.Row():
        ask_button = gr.Button("Ask")
        clear_button = gr.Button("Clear")
    
    answer_output = gr.Textbox(label="Answer", lines=5)
    sources_output = gr.Textbox(label="Retrieved Sources", lines=10)
    
    ask_button.click(
        fn=chat_interface,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    clear_button.click(
        fn=clear_conversation,
        inputs=None,
        outputs=[question_input, answer_output, sources_output]
    )

# Launch app
demo.launch(share=False, debug=True)
print("Gradio interface launched. Take a screenshot for the final report!")
