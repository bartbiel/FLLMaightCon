from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import torch
import sys
import re

sys.path.append(str(Path(__file__).resolve().parent.parent))
import grants

# ============ Load Model & Tokenizer ============
Mistral_snapshot = Path(grants.Mistral_snapshot)

tokenizer = AutoTokenizer.from_pretrained(
    Mistral_snapshot,
    local_files_only=True,
    trust_remote_code=True,
    use_fast=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    Mistral_snapshot,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

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

# ============ Load FAISS VectorStore ============
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory = "./data/faiss_index"

vectorstore = FAISS.load_local(
    persist_directory,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)

# ============ Field Extraction ============
def extract_airport_fields(text: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    code = name = latitude = longitude = None

    code_match = re.search(r"city_code:\s*([^\s,]+)", text, re.IGNORECASE)
    name_match = re.search(r"name:\s*([^\n\r,]+)", text, re.IGNORECASE)
    lat_match = re.search(r"latitude:\s*([^\s,]+)", text, re.IGNORECASE)
    lon_match = re.search(r"longitude:\s*([^\s,]+)", text, re.IGNORECASE)

    if code_match:
        code = code_match.group(1).strip()
    if name_match:
        name = name_match.group(1).strip()
    if lat_match:
        latitude = lat_match.group(1).strip()
    if lon_match:
        longitude = lon_match.group(1).strip()

    return code, name, latitude, longitude

# ============ Main Function to Return Selected Info ============
def get_airport_info(query: str) -> Optional[Dict[str, str]]:
    docs = vectorstore.similarity_search(query, k=10)
    candidates = []

    for doc in docs:
        code, name, lat, lon = extract_airport_fields(doc.page_content)
        if all([code, name, lat, lon]):
            candidates.append({
                "code": code,
                "name": name,
                "longitude": lon,
                "latitude": lat
            })

    if not candidates:
        print("[INFO] No matching results found.")
        return None

    print("\nSelect an airport:")
    for i, c in enumerate(candidates, 1):
        print(f"{i}. {c['name']} ({c['code']})")

    try:
        selection = input("Enter number: ").strip()
        index = int(selection) - 1
        if 0 <= index < len(candidates):
            return candidates[index]
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Please enter a valid number.")
        return None

# ============ CLI for Manual Testing ============
if __name__ == "__main__":
    print("=== Airport Lookup ===")
    while True:
        query = input("\nEnter airport or city name (or 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            break

        info = get_airport_info(query)
        if info:
            print("\n=== Selected Airport Info ===")
            print(f"Code:      {info['code']}")
            print(f"Name:      {info['name']}")
            print(f"Longitude: {info['longitude']}")
            print(f"Latitude:  {info['latitude']}")
