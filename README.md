# VectorMind-Chat ✨

**Intelligent Complaint Analysis for Financial Services**

---

## 📅 Project Overview

VectorMind-Chat is a Retrieval-Augmented Generation (RAG)-powered chatbot designed for **CrediTrust Financial**, a digital finance company in East Africa. This solution transforms millions of unstructured customer complaints from the Consumer Financial Protection Bureau (CFPB) into **actionable insights** for internal teams like **Product**, **Support**, and **Compliance**.

It supports seamless querying of complaints across five core product categories:

* ✨ Credit Cards
* ✨ Personal Loans
* ✨ Buy Now, Pay Later (BNPL)
* ✨ Savings Accounts
* ✨ Money Transfers

---

## 🌟 Business Objectives

| KPI     | Objective                                                |
| ------- | -------------------------------------------------------- |
| ✅ KPI 1 | Reduce time-to-insight from days to minutes              |
| ✅ KPI 2 | Empower Support & Compliance teams to self-serve queries |
| ✅ KPI 3 | Enable proactive issue detection via real-time feedback  |

---

## 📂 Repository Structure

```
VectorMind-Chat/
├── notebooks/                # EDA and preprocessing notebooks
├── src/                      # RAG pipeline and chatbot scripts
├── vector_store/             # FAISS or ChromaDB vector index
├── requirements.txt          # Dependencies
└── README.md                 # Project overview and setup
```

---

## ⚖️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd VectorMind-Chat
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### `requirements.txt` includes:

* `pandas`, `numpy`, `matplotlib`, `seaborn`
* `langchain`, `sentence-transformers`, `chromadb`
* `gradio`, `huggingface_hub`, `regex`, `python-dotenv`

---

## 📊 Dataset

* Download from [CFPB Consumer Complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/)
* File path: `data/complaints.csv`
* Size: \~5.64 GB

> If using **Google Colab**, mount Google Drive and update paths:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## ✍️ 1: EDA & Preprocessing

### Key Actions

* Loaded and processed \~9.6M complaint records in **chunks** to manage memory.
* **Filtered** for the 5 target product categories.
* Cleaned and normalized the `Consumer complaint narrative` for quality embeddings.

### Deliverables

* Notebook: `notebooks/eda_preprocessing.ipynb`
* Cleaned CSV: `data/filter/filtered_complaints.csv`
* Visualizations: Product distribution, length histograms

---
## ✍️ 2: Text Chunking, Embedding, and Vector Store Indexing

### 📌 Overview

Processed 470,118 rows to chunk complaint narratives, generate vector embeddings, and index them in ChromaDB, using iterative batching to handle 4GB RAM limitations.

### 🛠️ Key Activities

* Chunking: Used RecursiveCharacterTextSplitter with chunk_size=300 and chunk_overlap=50; tested chunk sizes (200, 300, 500) for trade-offs in context retention.

* Embedding: Leveraged all-MiniLM-L6-v2, embedding in batches of 50 using HuggingFaceEmbeddings.

* Indexing: Stored in ChromaDB with normalized metadata:

* complaint_id, product_category, issue, company, date_received, source: cfpb

* Validation: Queried: “What are common issues with BNPL?” — results showed strong text + metadata alignment.

### 💡 Key Insights

* chunk_size=300 balances semantic coherence with retrieval granularity.

* Iterative embedding strategy works within 4GB RAM constraints.

* Metadata design supports traceability (KPI 2 compliance).

📦 Deliverables

* 📓 Notebook: `notebooks/chunk_embed_index.ipynb`

* 🧠 Vector Store: `vector_store/chroma_db`

* 📊 Visualization: `vector_store/chunk_size_experiment.png`


# 3: Retrieval-Augmented Generation (RAG) Core Logic and Evaluation

## Overview

This task implements a Retrieval-Augmented Generation (RAG) pipeline to enhance customer complaint analysis for CrediTrust. The system retrieves relevant complaint excerpts and generates context-aware, professional answers using the Mistral-7B-Instruct-v0.2 model via Hugging Face’s Inference API.

The pipeline is evaluated on a set of representative questions, demonstrating retrieval effectiveness and answer quality.

---

## Key Activities

- **Retriever**  
  Utilizes the `sentence-transformers/all-MiniLM-L6-v2` model on GPU to embed user queries. Retrieves the top 5 most relevant complaint chunks from a ChromaDB vector store.

- **Prompt Engineering**  
  Designed a targeted prompt template instructing Mistral-7B to act as a financial analyst for CrediTrust, providing answers strictly based on the retrieved complaint context.

- **Generator**  
  Combines retrieved text chunks and user questions into the prompt, generating concise, accurate answers through Hugging Face’s Inference API.

- **Evaluation**  
  Tested the pipeline with 5 representative questions (e.g., “What are common issues with Buy Now, Pay Later (BNPL) services?”). Produced a detailed evaluation table including:  
  - Question  
  - Generated Answer  
  - Retrieved Sources  
  - Quality Score (1 to 5)  
  - Comments and Analysis  

- **Optimization**  
  Leveraged GPU acceleration for embedding computations and implemented memory management best practices to ensure smooth operation within Google Colab’s free tier constraints.

---

## Key Insights

- The retrieval mechanism is effective for product-specific queries such as BNPL and credit cards, ensuring relevant context is supplied to the model.

- Mistral-7B generates concise, professional, and contextually grounded responses, although answer detail may depend on the quality and quantity of retrieved information.

- Evaluation results identify opportunities for prompt refinement and highlight key performance indicators (KPIs) related to answer relevance and contextual accuracy.

---

## Deliverables

- **Script:** `src/rag_pipeline.py` — Complete RAG pipeline implementation  
- **Evaluation Report:** `vector_store/task3_evaluation.md` — Detailed question-answer evaluation table with scoring and analysis



## 🔗 References

* [CFPB Consumer Complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/)
* [LangChain Docs](https://docs.langchain.com/)
* [ChromaDB](https://docs.trychroma.com/)
* [Gradio](https://www.gradio.app/)
* [Sentence-Transformers](https://www.sbert.net/)

---
