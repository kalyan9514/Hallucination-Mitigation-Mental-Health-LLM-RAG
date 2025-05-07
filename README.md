# Enhancing Mental Health Information Retrieval with LLMs and RAG

**Overview:**

This project introduces a multi-phase AI pipeline to improve mental health information retrieval using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). It addresses hallucination reduction and improves factual consistency in sensitive mental health guidance. The system leverages models like Gemma, GPT-3.5/4, and RAG, tested on a 50,000+ record mental health dataset for condition classification and empathetic response generation.

**Dataset Summary:**

| Dataset Source               | Description                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| Curated Mental Health Corpus | Aggregated from multiple Kaggle datasets (e.g., Depression Reddit, Suicidal Tweet Detection) |
| Size                         | 50,000+ annotated mental health statements                                                   |
| Classes                      | Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder                         |

**Key Columns:**

| Column Name        | Description                                    |
| ------------------ | ---------------------------------------------- |
| `statement`        | User-submitted mental health-related statement |
| `processed_status` | Cleaned and labeled mental health condition    |
| `processed_text`   | Normalized and tokenized statement text        |

**System Architecture:**

The model pipeline is structured in 4 phases:

| Phase   | Description                                                         |
| ------- | ------------------------------------------------------------------- |
| Phase 1 | Baseline classification using Gemma                                 |
| Phase 2 | Reasoning-based refinement using GPT-4/GPT-3.5                      |
| Phase 3 | Reasoning + classification using Gemma                              |
| Phase 4 | Final prediction using Gemma + Retrieval-Augmented Generation (RAG) |

**Model Performance Comparison:**

| Phase                           | Accuracy   | Hallucination Rate |
| ------------------------------- | ---------- | ------------------ |
| Phase 1 – Gemma (Baseline)      | 42.00%     | 35.2%              |
| Phase 3 – Gemma + GPT Reasoning | 61.00%     | 15.2%              |
| **Phase 4 – Gemma + RAG**       | **83.00%** | **5.2%**           |

**How to Run the Project:**

1. Clone the Repository

```
git clone https://github.com/your-username/MentalHealth-RAG-LLM.git
cd MentalHealth-RAG-LLM
```

2. Install Required Libraries

```
pip install pandas numpy matplotlib seaborn transformers openai sentence-transformers scikit-learn
```

3. Run the Notebook

```
jupyter notebook Final_Code.ipynb
```

**Use Cases:**

- Personalized mental health support via chatbot systems

- Factual and context-aware answers grounded in mental health literature

- Reduced hallucinations in LLM-generated responses

- Enhanced empathy in responses using LLM+RAG hybrid reasoning

**Contact:**

For queries or suggestions, feel free to reach out to me on LinkedIn - https://www.linkedin.com/in/kalyan-kumar-8170a111b/







