from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import torch
from pathlib import Path
import sys

# Adjust this import according to your actual project structure
sys.path.append(str(Path(__file__).resolve().parent.parent))
import grants

def extract_city_code_and_name(doc_text):
    city_code, name = None, None
    lines = doc_text.split("\n")
    for line in lines:
        if "CityCode" in line:
            city_code = line.split(":", 1)[1].strip()
        elif "Name" in line:
            name = line.split(":", 1)[1].strip()
    return city_code, name

def main():
    # Load tokenizer and model snapshot path
    Mistral_snapshot = Path(grants.Mistral_snapshot)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Mistral_snapshot,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        Mistral_snapshot,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # Setup HuggingFace text generation pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1500,
        max_length=1024,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        truncation=True
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # Load embeddings model used for FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS index
    persist_directory = "./data/faiss_index"
    vectorstore = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # === DEBUG: Check example documents ===
    print("=== Debug: Checking example documents from FAISS index ===")
    example_docs = vectorstore.similarity_search("airport", k=5)
    print(f"Found {len(example_docs)} documents.\n")

    for i, doc in enumerate(example_docs, 1):
        print(f"--- Document {i} ---")
        print(doc.page_content)
        print("\n")

if __name__ == "__main__":
    main()
