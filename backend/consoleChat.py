from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from pathlib import Path
import torch
import sys
import re  # Added for regex parsing

sys.path.append(str(Path(__file__).resolve().parent.parent))
import grants

# Load tokenizer and model
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

# Load embeddings model for FAISS retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load saved FAISS index with safe deserialization
persist_directory = "./data/faiss_index"
vectorstore = FAISS.load_local(
    persist_directory,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create conversational retrieval chain with memory
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True  # source docs used internally by LLM
)


def extract_city_code_and_name(doc_text):
    city_code = None
    name = None

    # Regex to extract city_code and name, case insensitive
    city_code_match = re.search(r"city_code:\s*([^\s]+)", doc_text, re.IGNORECASE)
    name_match = re.search(r"name:\s*([^\n\r]+?)(?:\s+\w+?:|$)", doc_text, re.IGNORECASE)

    if city_code_match:
        city_code = city_code_match.group(1).strip()

    if name_match:
        name = name_match.group(1).strip()

    return city_code, name


print("=== Airport Info Retrieval ===")
print("Type 'exit' anytime to quit.\n")

while True:
    # Step 1: User provides city or airport name
    user_input = input("Enter city or airport name: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Exiting.")
        break

    # Step 2: Query FAISS for possible matches
    candidate_docs = vectorstore.similarity_search(user_input, k=10)

    # DEBUG: Show number of docs retrieved and snippet
    print(f"\n[DEBUG] Retrieved {len(candidate_docs)} docs for query '{user_input}':")
    for i, doc in enumerate(candidate_docs, 1):
        print(f"Doc {i} preview: {doc.page_content[:150].replace(chr(10), ' ')}...")

    # Extract city_codes and names, filter by partial match on name
    candidates = []
    for doc in candidate_docs:
        city_code, name = extract_city_code_and_name(doc.page_content)
        print(f"[DEBUG] Extracted city_code: {city_code}, name: {name}")
        if city_code and name and user_input.lower() in name.lower():
            if (city_code, name) not in candidates:
                candidates.append((city_code, name))
                print(f"[DEBUG] Added candidate: {name} ({city_code})")

    if not candidates:
        print("No matching airports found. Try again.\n")
        continue

    # Show candidate airports for selection
    print("\nSelect an airport by number:")
    for i, (code, name) in enumerate(candidates, 1):
        print(f"{i}. {name} ({code})")

    # Step 3: User selects airport
    selection = input("\nEnter number of the airport: ").strip()
    if selection.lower() in ["exit", "quit"]:
        print("Exiting.")
        break

    if not selection.isdigit() or not (1 <= int(selection) <= len(candidates)):
        print("Invalid selection. Try again.\n")
        continue

    selected_code = candidates[int(selection) - 1][0]

    # Step 4: Retrieve detailed info for selected city_code
    query = f"Airport information for city code {selected_code}"
    result = rag_chain({"question": query})

    print("\n=== Airport Info ===")
    print(result["answer"])  # Only print the answer text

    print("\n---\n")
