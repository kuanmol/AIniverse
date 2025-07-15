from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


def generate_answer(question, context, max_new_tokens=300):
    """Generate an answer using the LLM with a report-specific prompt."""
    if not context.strip():
        return "No context available to answer the question."

    prompt = f"""You are a helpful assistant with expertise in analyzing structured documents like reports, resumes,
     and business plans.

Use only the context provided below to answer the question. If the question asks for a list, like a table of contents,
 list all relevant items found in the context. Otherwise, answer clearly and concisely based on the details available.

Do not guess or hallucinate. If the context doesn't contain the answer, say "I don't know."
Context:
{context}

Question:
{question}

Answer:"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_k=40,
            do_sample=False
        )
        answer = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        logging.info("Answer generated successfully")
        return answer
    except Exception as e:
        logging.error(f"Answer generation failed: {str(e)}")
        return "Error generating answer."