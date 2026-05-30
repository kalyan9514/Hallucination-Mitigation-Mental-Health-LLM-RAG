# Hallucination Mitigation Mental Health LLM-RAG

A production-grade mental health diagnosis system that uses a multi-phase LLM pipeline
to reduce hallucinations and improve factual consistency in AI-generated mental health guidance.
The final system combines Gemma-2 9B with retrieval-augmented generation, achieving 83% accuracy
and a 5.2% hallucination rate on a 50,000+ record mental health dataset.

---

## Architecture

```
User Input (text)
   ↓
RAG Retriever (FAISS + BGE embeddings)
   ↓
Context Assembly (top-5 chunks)
   ↓
Gemma-2 9B LLM (diagnosis + response generation)
   ↓
Response Parser (disorder extraction)
   ↓
Gradio UI
```

---

## Research Pipeline

The production app is the result of a 4-phase research experiment:

| Phase | Approach | Accuracy | Hallucination Rate |
|-------|----------|----------|--------------------|
| Phase 1 | Gemma-2 9B baseline (no RAG) | 42% | 35.2% |
| Phase 2 | GPT-4 / GPT-3.5 classification with reasoning | 61% | 15.2% |
| Phase 3 | Ensemble voting across LLMs | 70.6% | 12.4% |
| Phase 4 | Gemma-2 9B + RAG (production) | 83% | 5.2% |

All research notebooks are preserved in `notebooks/` for reproducibility.

---

## Features

- RAG pipeline using FAISS index and BGE embeddings for context retrieval
- Gemma-2 9B LLM for diagnosis, symptom matching, and treatment suggestions
- Supports 7 disorder classifications: Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder
- Response parser with regex and fallback label matching
- Gradio UI for text-based queries
- GitHub Actions CI on every push

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Gemma-2 9B (google/gemma-2-9b-it) |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Retrieval | FAISS |
| UI | Gradio 4.x |
| Containerisation | Docker |
| CI/CD | GitHub Actions |

---

## Prerequisites

- Python 3.11+
- Docker Desktop
- Hugging Face account with access to Gemma-2

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/kalyan9514/Hallucination-Mitigation-Mental-Health-LLM-RAG.git
cd Hallucination-Mitigation-Mental-Health-LLM-RAG
```

**2. Create your .env file**
```bash
cp .env.example .env
```

**3. Add your Hugging Face token to .env**
```bash
HUGGINGFACE_TOKEN=your_token_here
```

**4. Start with Docker**
```bash
docker compose up -d
```

Or run directly:

**5. Run the Gradio app**
```bash
python cmd/gradio/main.py
```

---

## Services

| Service | URL |
|---------|-----|
| Gradio diagnosis assistant | http://localhost:7860 |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| HUGGINGFACE_TOKEN | Token for loading Gemma-2 and other gated models |
| LLM_MODEL_ID | LLM model identifier (default: google/gemma-2-9b-it) |
| EMBEDDING_MODEL_ID | Embedding model (default: BAAI/bge-base-en-v1.5) |
| RAG_TOP_K | Number of chunks to retrieve (default: 5) |
| MAX_NEW_TOKENS | Max tokens for LLM generation (default: 500) |
| TEMPERATURE | Sampling temperature (default: 0.2) |

---

## Project Structure

```
├── cmd/
│   └── gradio/             # Gradio diagnosis app entry point
├── internal/
│   ├── rag/                # FAISS retriever and knowledge base loader
│   └── diagnosis/          # Engine and response parser
├── config/
│   └── config.py           # Pydantic settings and path constants
├── notebooks/
│   ├── eda.ipynb           # Exploratory data analysis
│   ├── phase_1.ipynb       # Baseline Gemma classification
│   ├── phase_2.ipynb       # GPT-4 / GPT-3.5 reasoning
│   ├── phase_3.ipynb       # Ensemble voting across LLMs
│   ├── phase_4.ipynb       # Gemma + RAG (production approach)
│   └── evaluation.ipynb    # Accuracy and F1 comparison across phases
├── data/                   # FAISS index and knowledge base (not committed)
├── logs/                   # Runtime logs (not committed)
├── tests/                  # Unit tests
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI
├── Dockerfile              # Docker image for Gradio app
├── docker-compose.yml      # Local service setup
├── DECISIONS.md            # Architecture decision records
├── .env.example            # Safe credentials template
└── requirements.txt        # Python dependencies
```

---

## Contact

[LinkedIn — Kalyan Kumar Chenchu Malakondaiah](https://www.linkedin.com/in/kalyan-kumar-8170a111b/)