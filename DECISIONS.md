# Architecture Decision Records

This document explains the key design decisions made during the refactor
of the Hallucination Mitigation Mental Health LLM-RAG project from a
Colab notebook collection into a production-grade project.

---

## 1. Modular `internal/` package structure

**Decision:** Split all core logic into separate modules under `internal/`.

**Why:** The original code had everything in a single flat `app.py` with
hardcoded Google Drive paths. Each module now has a single responsibility,
making it testable and reusable independently.

---

## 2. Notebooks preserved in `notebooks/` folder

**Decision:** All research phase notebooks are kept in `notebooks/` rather
than deleted.

**Why:** The notebooks represent the research experiments that led to the
final RAG approach. They document the progression from Phase 1 baseline
(40% accuracy) through Phase 4 RAG (83% accuracy) and serve as a reference
for the design decisions behind the production pipeline.

---

## 3. Production app uses Phase 4 pipeline only

**Decision:** The Gradio app implements only the Phase 4 Gemma + RAG pipeline.

**Why:** Phase 4 achieved the best results (83% accuracy, 5.2% hallucination
rate). Phases 1 through 3 are research experiments and are not suitable for
a user-facing app. The notebooks remain available for reproducibility.

---

## 4. Centralized config with Pydantic Settings

**Decision:** All environment variables and constants live in `config/config.py`.

**Why:** The original app had hardcoded API tokens, Google Drive paths, and
model names scattered throughout the code. Pydantic Settings validates types
at startup and loads from `.env` automatically.

---

## 5. Single Dockerfile

**Decision:** One Dockerfile for the single Gradio app.

**Why:** Unlike SafeSpace which had two separate interfaces, this project has
a single entry point. A single Dockerfile keeps the setup simple.

---

## 6. `refactor/production-structure` branch strategy

**Decision:** All refactor work is done on a feature branch, not directly on `main`.

**Why:** Keeps `main` stable while the refactor is in progress. Once reviewed,
the branch gets merged via a pull request.