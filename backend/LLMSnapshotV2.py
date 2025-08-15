# LLMSnapshot.py

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
import grants


class FlightSummaryLLM:
    def __init__(self):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading LLM...")

        # === Load LLM ===
        mistral_snapshot = Path(grants.Mistral_snapshot)

        tokenizer = AutoTokenizer.from_pretrained(
            mistral_snapshot,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            mistral_snapshot,
            local_files_only=True,
            torch_dtype=torch.float32,  # keep FP32
            device_map="cpu",
            low_cpu_mem_usage=True
             
        )

        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            
        )

        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        # === Load FAISS retriever ===
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        persist_directory = "./data/faiss_index"
        vectorstore = FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",  # only store the answer in memory
            return_messages=True
        )

        # === Setup RAG Chain ===
        self.rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] LLM Ready ✅")

    def ask(self, constant_data, floating_data):
        start_time = datetime.now()
        print(f"[{start_time.strftime('%H:%M:%S')}] Generating summary...")

        user_query = f"""
        You are an expert in airport selection based on flight data.
        For flight data {constant_data}, summarize changes of {floating_data} over time.
        Provide a detailed analysis of the flight's trajectory, speed, altitude, and any other relevant parameters.
        Summarize in 5 sentences.
        """

        result = self.rag_chain.invoke({"question": user_query})

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        print(f"[{end_time.strftime('%H:%M:%S')}] Summary generated in {elapsed:.2f}s ✅")

        return result["answer"]
