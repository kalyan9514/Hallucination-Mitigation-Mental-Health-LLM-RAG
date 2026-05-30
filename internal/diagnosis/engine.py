"""
internal/diagnosis/engine.py

Core diagnosis engine that ties together the retriever,
LLM generation, and response parsing.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config.config import settings
from internal.rag.retriever import Retriever
from internal.diagnosis.parser import extract_diagnosis, clean_response

logger = logging.getLogger(__name__)


class DiagnosisEngine:
    """
    Loads the LLM and runs the full RAG-based diagnosis pipeline.
    Instantiate once and reuse across requests.
    """

    def __init__(self):
        logger.info("Loading tokenizer and LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.llm_model_id,
            token=settings.huggingface_token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model_id,
            token=settings.huggingface_token,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.generator = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=settings.temperature,
            repetition_penalty=settings.repetition_penalty,
            return_full_text=False,
            max_new_tokens=settings.max_new_tokens,
        )
        self.retriever = Retriever()
        logger.info("Diagnosis engine ready.")

    def run(self, query: str) -> dict:
        """
        Run the full RAG diagnosis pipeline for a given user query.

        Returns a dict with the response text and extracted diagnosis label.
        """
        # Retrieve relevant context chunks
        top_chunks = self.retriever.search(query)
        context = "\n".join([f"Doc {i}:\n{chunk}" for i, chunk in enumerate(top_chunks)])

        # Build the prompt using the same format as the original app
        prompt = self._build_prompt(query, context)
        full_prompt = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        # Generate response
        raw_response = self.generator(full_prompt)[0]["generated_text"]
        response = clean_response(raw_response)
        diagnosis = extract_diagnosis(response)

        return {
            "response": response,
            "diagnosis": diagnosis,
            "context_chunks": top_chunks,
        }

    def _build_prompt(self, query: str, context: str) -> list[dict]:
        """Build the chat prompt from the user query and retrieved context."""
        labels = ", ".join(settings.disorder_labels)
        return [
            {
                "role": "user",
                "content": (
                    f"You are an AI assistant trained to identify mental health conditions.\n"
                    f"Use the context to diagnose and respond to the following query.\n"
                    f"Choose only one **Diagnosed Mental Disorder** from: [{labels}]\n\n"
                    f"Return:\n"
                    f"1. **Diagnosed Mental Disorder**\n"
                    f"2. **Matching Symptoms**\n"
                    f"3. **Personalized Treatment**\n"
                    f"4. **Helpline Numbers**\n"
                    f"5. **Source Link**\n\n"
                    f"If you cannot determine the disorder, output "
                    f"'Diagnosed Mental Disorder: Unknown'.\n\n"
                    f"Retrieved Context:\n{context}\n\n"
                    f"User Query: {query}"
                ),
            }
        ]
    