
import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

BASE_DIR = Path(__file__).resolve().parent.parent

# Load university knowledge
with open(BASE_DIR / "documents" / "university_knowledge.txt",
          "r", encoding="utf-8") as file:
    knowledge = file.read()

chunks = [
    text.strip() for text in knowledge.split("\n\n")
    if text.strip()
]

# Load RAG model and index
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load fine-tuned model
model_path = BASE_DIR / "fine_tuning" / "university_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Evaluation questions
questions = [
    "What is the minimum attendance required?",
    "Which courses does the university offer?",
    "What resources are available in the library?",
    "What examinations must students attend?",
    "How can students apply for admission?"
]

print("\n========== RAG VS FINE-TUNING ==========")

for question in questions:
    print("\nQuestion:", question)

    # RAG retrieval
    start = time.time()

    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, results = index.search(query_embedding, k=1)

    rag_answer = chunks[results[0][0]]

    rag_time = time.time() - start

    # Fine-tuning model
    start = time.time()

    prompt = question

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_beams=4,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3
    )

    fine_tuned_answer = tokenizer.decode(
        outputs[0], skip_special_tokens=True
    )

    fine_time = time.time() - start

    print("\nRAG Retrieved Information:")
    print(rag_answer)

    print("\nFine-Tuned Model Answer:")
    print(fine_tuned_answer)

    print("\nRAG Time:", round(rag_time, 4), "seconds")
    print("Fine-Tuning Model Time:", round(fine_time, 4), "seconds")

print("\n========== COMPARISON COMPLETED ==========")
