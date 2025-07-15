# # llm_answer.py
#
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# import torch
#
# # Load FLAN-T5 once
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_name = "google/flan-t5-base"
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
#
# def generate_answer(question, context, max_new_tokens=128):
#     prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
#
#     output = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         temperature=0.7,
#         top_k=50,
#         do_sample=False
#     )
#
#     return tokenizer.decode(output[0], skip_special_tokens=True).strip()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def generate_answer(question, context, max_new_tokens=128):
    prompt = f"""You are reading a resume. Use only the context below to answer the question.
If the answer is not found in the context, say "I don't know."

Context:
{context}

Question:
{question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_k=50,
        do_sample=False
    )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()
