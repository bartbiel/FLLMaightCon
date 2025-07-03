from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pathlib import Path
import torch
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import grants



# ==================== Load Model and Tokenizer ====================
model_path = Path(grants.MistralDIR)
Mistral_snapshot=Path(grants.Mistral_snapshot)

tokenizer = AutoTokenizer.from_pretrained(
    Mistral_snapshot,
    local_files_only=True,
    trust_remote_code=True,
    use_fast=False  # force slow version
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Configured padding token using EOS token")

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

# ==================== Load data from CSV ====================
csv_loader = CSVLoader(file_path="./data/airports.csv",
                       csv_args={"delimiter": ","},
                       encoding="utf-8")
csv_docs = csv_loader.load()


# ==================== Split Documents into Chunks ====================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(csv_docs)

# ==================== Embed and Index ====================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory = "./data/faiss_index"

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

# Save the database to disk
vectorstore.save_local(persist_directory)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==================== Create RAG Chain ====================
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ==================== Example Query ====================
query = "Provide the airport data for Lublin"
response = rag_chain.invoke({"query": query})

print("====== RAG Answer ======")
print(response["result"])

print("\n====== Source Documents Used ======")
for doc in response["source_documents"]:
    print(f"- Excerpt: {doc.page_content[:200]}...\n")
