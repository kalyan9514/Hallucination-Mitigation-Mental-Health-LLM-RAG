
import os
import torch
import faiss
import numpy as np
import gradio as gr
from google.colab import drive
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
hf_auth_token = os.getenv('HF_TOKEN')
print(hf_auth_token)  # Debugging token load

# Mount Google Drive
drive.mount('/content/drive')

# Define paths
embedding_model_name = "BAAI/bge-base-en-v1.5"
embedding_model_local_dir = "/content/drive/MyDrive/models/bge-base-en-v1.5"
faiss_index_file = "/content/qa_faiss_embedding.index"
text_chunks_file = "/content/chunked_text_RAG_text.txt"
llm_model_name = "google/gemma-2-9b-it"

# Create directory if not exists
os.makedirs(embedding_model_local_dir, exist_ok=True)

# Load embedding model (SentenceTransformer)
if not os.path.exists(os.path.join(embedding_model_local_dir, "config.json")):
    print("Saving embedding model to Google Drive...")
    vector_model = SentenceTransformer(embedding_model_name)
    vector_model.save(embedding_model_local_dir)
    print("Model saved successfully!")
else:
    print("Loading embedding model from Google Drive...")
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    vector_model = SentenceTransformer(embedding_model_local_dir, device=device_type)

# Load FAISS index
faiss_index = faiss.read_index(faiss_index_file)
print("FAISS index loaded!")

# Load text chunks for retrieval
def read_text_chunks():
    with open(text_chunks_file, "r", encoding="utf-8") as file:
        return file.read().split("\n\n---\n\n")

retrieved_chunks = read_text_chunks()
print(f"{len(retrieved_chunks)} document chunks loaded.")

# Load LLM
device_id = 0 if torch.cuda.is_available() else -1
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device_id)
response_generator = pipeline(
    model=llm_model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
    device=device_id,
)

# Query processor
def generate_diagnosis(user_input):
    # Embed user query
    user_embedding = vector_model.encode(user_input, normalize_embeddings=True)
    user_embedding = np.array([user_embedding], dtype=np.float32)

    # Search top 5 matches from FAISS
    _, top_indices = faiss_index.search(user_embedding, 5)
    relevant_docs = [retrieved_chunks[i] for i in top_indices[0]]

    # Construct retrieval context
    context_text = "\nRetrieved Context:\n" + "".join([f"Doc {i}:\n{doc}\n" for i, doc in enumerate(relevant_docs)])

    # Define the prompt
    chat_prompt = [
        {"role": "user", "content": f"""
You are an AI assistant trained to identify mental health conditions.

Use the context to diagnose and respond to the following query.

Choose only one **Diagnosed Mental Disorder** from:
[Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, Personality Disorder]

Return:
1. **Diagnosed Mental Disorder**
2. **Matching Symptoms**
3. **Personalized Treatment**
4. **Helpline Numbers**
5. **Source Link**

If you cannot determine the disorder, output "Diagnosed Mental Disorder: Unknown".

---
Context:
{context_text}

User Query: {user_input}
"""}
    ]

    full_prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
    ai_response = response_generator(full_prompt)[0]["generated_text"]
    return ai_response

# Gradio interface
gradio_interface = gr.Interface(
    fn=generate_diagnosis,
    inputs=gr.Textbox(lines=2, placeholder="Describe your emotional state..."),
    outputs=gr.Textbox(label="AI Mental Health Diagnosis"),
    title="🧠 Mental Health Assistant",
    description="Enter your concerns and receive an AI-generated diagnosis and support.",
)

# Launch app
gradio_interface.launch(share=True)
