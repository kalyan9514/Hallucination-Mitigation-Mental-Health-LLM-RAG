"""
cmd/gradio/main.py

Gradio app entry point for the Hallucination Mitigation Mental Health assistant.
Provides a simple text interface that runs the full RAG diagnosis pipeline.
"""

import logging
import gradio as gr
from internal.diagnosis.engine import DiagnosisEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the engine once at startup
engine = DiagnosisEngine()


def handle_query(user_input: str) -> tuple[str, str]:
    """
    Handle a text query through the diagnosis pipeline.
    Returns the full response and the extracted diagnosis label.
    """
    if not user_input.strip():
        return "Please enter a message.", "N/A"

    result = engine.run(user_input)
    return result["response"], result["diagnosis"]


def build_ui() -> gr.Blocks:
    """Build and return the Gradio interface."""
    with gr.Blocks(title="Mental Health Assistant") as app:
        gr.Markdown("# Mental Health Assistant")
        gr.Markdown(
            "Enter your concerns below and receive an AI-generated diagnosis and support. "
            "Powered by Gemma-2 9B with retrieval-augmented generation."
        )

        with gr.Row():
            user_input = gr.Textbox(
                lines=3,
                placeholder="Describe your emotional state or symptoms...",
                label="Your Query",
            )

        submit_btn = gr.Button("Submit")

        with gr.Row():
            response_out = gr.Textbox(label="AI Response", lines=10)
            diagnosis_out = gr.Textbox(label="Diagnosed Disorder", lines=1)

        submit_btn.click(
            handle_query,
            inputs=[user_input],
            outputs=[response_out, diagnosis_out],
        )

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=False)