# Enterprise Knowledge Intelligence Platform

A production-oriented **Retrieval-Augmented Generation (RAG)** platform for enterprise document search and grounded question answering.

The platform ingests enterprise documents, extracts and chunks their content, generates semantic embeddings, stores them in PostgreSQL with `pgvector`, retrieves relevant passages, reranks candidates with a cross-encoder, and generates grounded answers using a local LLM.

---

## 🚀 Project Overview

Enterprise teams often need to search across internal documents such as:

* HR policies
* Employee handbooks
* Company guidelines
* Operational documentation
* Internal procedures
* Knowledge-base documents

Traditional keyword search can struggle when users phrase questions differently from the source documents.

This project addresses that problem using a **two-stage retrieval pipeline**:

```text
User Question
      ↓
Query Embedding
      ↓
Dense Vector Retrieval
      ↓
Candidate Documents
      ↓
Cross-Encoder Reranking
      ↓
Relevant Context
      ↓
Local LLM
      ↓
Grounded Answer + Sources
```

The system is designed to answer questions using information contained in the indexed documents rather than relying on unsupported external knowledge.

---

# 🎬 Demo

The platform exposes an interactive FastAPI interface for document ingestion, semantic search, and grounded question answering.

## API Documentation

![FastAPI API Documentation](docs/screenshots/api-docs.png)

## Semantic Search

The semantic search endpoint retrieves the most relevant document chunks using vector similarity search.

![Semantic Search](docs/screenshots/search.png)

## Grounded RAG Answer

The `/ask` endpoint combines dense retrieval, cross-encoder reranking, and local LLM generation to produce a grounded answer with supporting sources.

![RAG Answer](docs/screenshots/rag-answer.png)


# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │   Enterprise Docs    │
                         │ PDF / DOCX / TXT     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Document Extraction  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Text Chunking     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Sentence Transformer │
                         │   384-dim Embedding  │
                         └──────────┬───────────┘
                                    │
                                    ▼
              ┌─────────────────────────────────────────┐
              │          PostgreSQL + pgvector          │
              │                                         │
              │ Documents │ Chunks │ Metadata │ Vectors│
              └────────────────────┬────────────────────┘
                                   │
                                   │ Dense Retrieval
                                   ▼
                         ┌──────────────────────┐
                         │ Candidate Chunks     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Cross-Encoder        │
                         │ Reranking            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Relevant Context     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Local LLM            │
                         │ Llama 3.2 3B         │
                         │ via Ollama            │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Grounded Answer      │
                         │ + Source Chunks      │
                         └──────────────────────┘
```

---

# ✨ Key Features

* PDF, DOCX, and TXT document ingestion
* Structured document extraction
* Text chunking for retrieval
* Sentence-Transformer embeddings
* 384-dimensional vector representations
* PostgreSQL + pgvector vector storage
* Cosine-distance similarity search
* Cross-encoder reranking
* Retrieval-Augmented Generation
* Local LLM inference using Ollama
* Grounded question answering
* Source-aware responses
* Duplicate document detection
* Document listing and deletion
* Database health checks
* Automated test suite
* Retrieval evaluation
* Reranking evaluation
* Threshold calibration

---

# 🧰 Technology Stack

| Layer                  | Technology                               |
| ---------------------- | ---------------------------------------- |
| Language               | Python 3.12                              |
| API                    | FastAPI                                  |
| Database               | PostgreSQL 16                            |
| Vector Database        | pgvector                                 |
| ORM                    | SQLAlchemy                               |
| Database Migrations    | Alembic                                  |
| Embedding Model        | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding Dimension    | 384                                      |
| Reranker               | `cross-encoder/ms-marco-MiniLM-L-6-v2`   |
| LLM                    | Llama 3.2 3B                             |
| LLM Runtime            | Ollama                                   |
| Cache / Infrastructure | Redis                                    |
| Testing                | pytest                                   |
| Containerization       | Docker / Docker Compose                  |

---

# 🔄 RAG Pipeline

## 1. Document Ingestion

The API accepts:

```text
.txt
.pdf
.docx
```

The uploaded document is validated and passed to the appropriate document extractor.

---

## 2. Document Extraction

The extraction layer converts supported documents into structured content blocks.

Metadata can include information such as:

* source filename
* page information
* document structure

This metadata is preserved through the ingestion pipeline.

---

## 3. Text Chunking

Extracted text is divided into smaller chunks.

Chunking improves retrieval because the vector search system can identify specific relevant passages instead of retrieving an entire document.

Each stored chunk contains:

```text
document_id
chunk_index
content
metadata
embedding
```

---

## 4. Embedding Generation

Each chunk is converted into a semantic embedding using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model produces:

```text
384-dimensional embeddings
```

Embeddings are normalized before being stored.

---

## 5. Vector Storage

Embeddings are stored directly in PostgreSQL using `pgvector`.

This allows the application to keep:

```text
Application Data
        +
Document Chunks
        +
Vector Embeddings
```

inside the same database system.

---

## 6. Dense Retrieval

When a user submits a question:

1. The question is converted into an embedding.
2. The embedding is compared with stored chunk embeddings.
3. The closest chunks are retrieved using cosine distance.
4. The retrieved chunks become candidates for reranking.

---

## 7. Cross-Encoder Reranking

Dense retrieval provides the initial candidate set.

The candidates are then scored using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder evaluates the relationship between:

```text
Question
   +
Candidate Chunk
```

and produces a relevance score.

The candidates are then sorted by this score.

This two-stage architecture separates:

```text
Fast Candidate Retrieval
          ↓
More Precise Relevance Ranking
```

---

## 8. Grounded LLM Generation

The highest-ranked relevant chunks are assembled into a context.

The local LLM receives:

```text
System Instructions
        +
Retrieved Context
        +
User Question
```

The model is instructed to:

* use only the supplied context
* avoid inventing facts
* avoid unsupported external knowledge
* state when the information is unavailable
* provide a concise answer

The final API response includes both the generated answer and the supporting source chunks.

---

# 📊 Evaluation

The project includes a retrieval evaluation dataset containing:

```text
34 total queries
18 positive queries
16 negative queries
```

The evaluation was designed to measure both:

1. Whether relevant documents can be retrieved.
2. Whether irrelevant questions can be rejected.

---

## Dense Retrieval Results

Initial dense retrieval produced:

| Metric      | Result |
| ----------- | -----: |
| Recall@1    |   1.00 |
| Recall@3    |   1.00 |
| MRR@3       |   1.00 |
| Precision@3 |   0.33 |

The results show that relevant documents were consistently retrieved near the top of the candidate list.

However, dense similarity alone produced too many false positives for negative queries at the initial threshold.

At a similarity threshold of `0.70`:

| Metric                  | Result |
| ----------------------- | -----: |
| Negative Query Accuracy |   0.19 |
| False Positive Rate     |   0.81 |

Threshold calibration on the evaluation dataset identified `0.4605` as a better balance:

| Metric              | Result |
| ------------------- | -----: |
| Positive Recall     |   0.78 |
| Negative Accuracy   |   0.75 |
| False Positive Rate |   0.25 |
| Overall Score       |   0.76 |

---

## Cross-Encoder Reranking Results

Reranking improved answerability discrimination on the evaluation dataset.

Using the calibrated threshold of `4.1838`:

| Metric              | Result |
| ------------------- | -----: |
| Positive Recall     |   0.83 |
| Negative Accuracy   |   0.88 |
| False Positive Rate |   0.12 |
| Overall Score       |   0.85 |

The reranker therefore provided better separation between answerable and non-answerable queries on this evaluation set.

> **Important:** These thresholds are evaluation-set calibration results, not universal production thresholds. A larger and more representative enterprise evaluation dataset would be required before treating them as production defaults.

---

# 🧪 Testing

The current automated test suite passes completely:

```text
34 passed
```

Run the test suite with:

```powershell
python -m pytest -q
```

The tests cover areas including:

* embedding generation
* text chunking
* document extraction
* document ingestion
* repository operations
* vector similarity search
* RAG behavior
* reranking
* document APIs
* upload APIs
* search APIs
* question-answering APIs

---

# 🔌 API

The application exposes the following main endpoints.

## Health

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "service": "enterprise-knowledge-intelligence"
}
```

---

## Database Health

```http
GET /health/database
```

Example:

```json
{
  "database": "connected",
  "result": 1
}
```

---

## Upload Document

```http
POST /documents/upload
```

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/upload" `
  -F "file=@hr_policy.txt"
```

---

## List Documents

```http
GET /documents
```

---

## Delete Document

```http
DELETE /documents/{document_id}
```

---

## Semantic Search

```http
POST /search
```

Request:

```json
{
  "query": "How many vacation days do employees receive?",
  "top_k": 3
}
```

The response contains ranked document chunks and their similarity distances.

---

## RAG Question Answering

```http
POST /ask
```

Request:

```json
{
  "question": "How many vacation days do employees receive?",
  "top_k": 3
}
```

Example response:

```text
Employees receive 20 paid vacation days per calendar year.
```

The response also includes the source chunks used to construct the answer.

---

# 🖥️ Running the Project Locally

## Prerequisites

Install:

* Python 3.12+
* Docker Desktop
* Ollama

The project uses Docker for PostgreSQL/pgvector and Redis.

---

## 1. Clone the Repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>

cd enterprise-knowledge-intelligence
```

Replace:

```text
<YOUR_GITHUB_REPOSITORY_URL>
```

with your actual GitHub repository URL.

---

## 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```text
DATABASE_URL=postgresql+psycopg://eki_user:eki_password@localhost:5432/eki_db
REDIS_URL=redis://localhost:6379/0

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

LLM_MODEL=llama3.2:3b
LLM_BASE_URL=http://localhost:11434
```

Do **not** commit `.env` or other files containing secrets.

---

## 5. Start Infrastructure

```powershell
docker compose up -d
```

This starts the required infrastructure services.

---

## 6. Run Database Migrations

```powershell
alembic upgrade head
```

The project uses Alembic migrations rather than creating database tables automatically from the ORM.

---

## 7. Start Ollama

Make sure Ollama is running and the configured model is available locally.

The default configuration expects:

```text
http://localhost:11434
```

and:

```text
llama3.2:3b
```

---

## 8. Start the FastAPI Application

```powershell
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 📁 Project Structure

```text
enterprise-knowledge-intelligence/
│
├── alembic/
│   ├── env.py
│   └── versions/
│
├── app/
│   ├── core/
│   │   ├── config.py
│   │   └── dependencies.py
│   │
│   ├── db/
│   │   └── session.py
│   │
│   ├── models/
│   │   └── document.py
│   │
│   ├── repositories/
│   │   └── document_repository.py
│   │
│   ├── schemas/
│   │   ├── document.py
│   │   └── search.py
│   │
│   ├── services/
│   │   ├── chunk_embedding_service.py
│   │   ├── chunking_service.py
│   │   ├── document_extractor.py
│   │   ├── document_ingestion_service.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   ├── rag_service.py
│   │   └── reranking_service.py
│   │
│   └── main.py
│
├── evaluation/
│   ├── retrieval_dataset.json
│   ├── retrieval_evaluator.py
│   └── reranking_evaluator.py
│
├── tests/
│   ├── conftest.py
│   ├── test_embeddings.py
│   ├── test_chunking.py
│   ├── test_chunk_embedding_service.py
│   ├── test_document_extractor.py
│   ├── test_document_ingestion_service.py
│   ├── test_document_repository.py
│   ├── test_similarity_search.py
│   ├── test_rag_service.py
│   ├── test_reranking_service.py
│   ├── test_search.py
│   ├── test_upload.py
│   ├── test_ask.py
│   └── test_documents.py
│
├── alembic.ini
├── docker-compose.yml
├── .env.example
└── README.md
```

---

# 🧠 Engineering Decisions

## Why PostgreSQL + pgvector?

PostgreSQL provides the application's relational data layer while `pgvector` adds vector similarity search.

This allows the system to manage:

```text
Documents
+
Document Chunks
+
Metadata
+
Embeddings
```

within the same database.

---

## Why Two-Stage Retrieval?

Dense vector retrieval is useful for semantic candidate generation but can return false positives.

The system therefore uses:

```text
Dense Retrieval
      ↓
Candidate Generation
      ↓
Cross-Encoder Reranking
      ↓
Relevant Context
```

This combines efficient candidate retrieval with a more precise relevance model.

---

## Why a Local LLM?

The project uses Ollama for local LLM inference.

Benefits include:

* local development
* no dependency on a paid external LLM API
* improved control over enterprise document data
* reproducible development environment

---

## Why Alembic?

Database schema changes are managed through explicit migrations.

This provides a safer and more reproducible approach than relying on automatic table creation.

---

## Why Automated Evaluation?

A RAG system should not be evaluated only by whether the API returns an answer.

This project evaluates:

* retrieval quality
* ranking quality
* positive query recall
* negative query rejection
* false-positive rate
* answerability discrimination

This makes the retrieval pipeline measurable rather than purely qualitative.

---

# 🔐 Grounding Strategy

The LLM is explicitly instructed to answer using only the retrieved context.

The generation layer follows these principles:

```text
Use retrieved context
        ↓
Do not invent facts
        ↓
Do not rely on outside knowledge
        ↓
If evidence is insufficient
        ↓
Say that the information is unavailable
```

The API also returns the source chunks used for the answer, making the generated response easier to inspect and validate.

---

# ⚠️ Current Limitations

This is a production-oriented portfolio implementation rather than a fully hardened enterprise deployment.

Current limitations include:

* authentication is not implemented
* authorization is not implemented
* document-level access control is not implemented
* ingestion is synchronous
* observability/tracing is limited
* evaluation dataset is relatively small
* threshold calibration is dataset-specific
* hybrid BM25 + vector retrieval is not implemented
* multi-tenant isolation is not implemented
* streaming LLM responses are not implemented

---

# 🔮 Future Improvements

Potential next improvements include:

* Hybrid BM25 + vector retrieval
* Async/background document ingestion
* Redis-backed task queues
* Authentication and authorization
* Document-level access control
* Multi-tenant isolation
* Structured logging
* Distributed tracing
* Retrieval observability
* Streaming LLM responses
* Larger evaluation datasets
* Automated evaluation in CI/CD
* Model/version tracking
* Chunking strategy experiments
* RAG quality monitoring
* Kubernetes deployment

---

# 📈 Project Validation

The implementation has been validated at both the automated-test and running-API levels.

```text
Automated Tests
34 passed

API Health
Passed

Database Health
Passed

Document Upload
Passed

Semantic Search
Passed

RAG Question Answering
Passed

Document Deletion
Passed

Temporary Smoke-Test Data
Cleaned Up
```

Example end-to-end flow:

```text
Upload Document
      ↓
Extract Text
      ↓
Chunk Text
      ↓
Generate Embeddings
      ↓
Store in PostgreSQL + pgvector
      ↓
Search
      ↓
Rerank
      ↓
Generate Answer
      ↓
Return Answer + Sources
```

---

# 🎯 What This Project Demonstrates

This project demonstrates practical experience with:

* Retrieval-Augmented Generation
* Semantic search
* Vector databases
* Embedding models
* Cross-encoder reranking
* Grounded LLM generation
* Document processing pipelines
* PostgreSQL
* pgvector
* FastAPI
* SQLAlchemy
* Alembic
* Docker
* Automated testing
* Retrieval evaluation
* Reranking evaluation
* Threshold calibration
* Local LLM inference
* Production-oriented GenAI architecture

---

# 📌 Project Status

**Core implementation complete.**

The current version has passed the automated test suite and a full running-API smoke test covering document ingestion, vector search, RAG question answering, and document deletion.

---

# 📄 License

This project is intended as a portfolio and learning project.

Add your preferred license here before publishing the repository.
