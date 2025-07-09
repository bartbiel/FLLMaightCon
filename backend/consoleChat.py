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
from typing import Optional, Dict, Tuple, List
#from getAirfieldData import extract_airport_fields

sys.path.append(str(Path(__file__).resolve().parent.parent))
import grants
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



def airport_selection(typeOf: str):
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


    result = "No airport selected"

    while True:
        user_input = input(f"Enter city or airport name for {typeOf} (or type 'exit'): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting.")
            return result

        # Step 1: search using FAISS
        candidate_docs = vectorstore.similarity_search(user_input, k=10)

        if not candidate_docs:
            print("No results found. Try again.\n")
            continue

        # Step 2: Extract candidate (code, name) pairs
        candidates = []
        for doc in candidate_docs:
            city_code, name = extract_city_code_and_name(doc.page_content)
            if city_code and name:
                candidates.append((city_code, name))

        if not candidates:
            print("No valid airport data found. Try again.\n")
            continue

        # Step 3: Present options to the user
        print("\nMatching airports:")
        for idx, (code, name) in enumerate(candidates, 1):
            print(f"{idx}. {name} ({code})")

        # Step 4: User selects from the list
        while True:
            selection = input("\nEnter number of the airport to select (or 'exit'): ").strip()
            if selection.lower() in ["exit", "quit"]:
                print("Exiting.")
                return result

            if not selection.isdigit():
                print("Please enter a number.")
                continue

            selection_num = int(selection)
            if 1 <= selection_num <= len(candidates):
                selected_doc = candidate_docs[selection_num - 1]
                code, name, latitude, longitude = extract_airport_fields(selected_doc.page_content)

                if all([code, name, latitude, longitude]):
                    print(f"\nSelected {typeOf} airport: {name} ({code})")
                    return {
                        "code": code,
                        "name": name,
                        "latitude": latitude,
                        "longitude": longitude
                    }
                else:
                    print("Incomplete airport data. Try again.")
                    break  # go back to main input loop
            else:
                print("Invalid number. Please choose from the list.")


def final_airport_data(typeOf: str):
    res = airport_selection(typeOf)
    #if res:
        #print(f"Selected {typeOf.capitalize()} Airport: {res['name']} ({res['code']}) at {res['latitude']}, {res['longitude']}")
    return res