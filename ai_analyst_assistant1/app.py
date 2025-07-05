import streamlit as st
import logging
import re

import pandas as pd
from data_loader import load_csv, get_data_summary, get_preview
from pdf_loader import extract_text_from_pdf
from summarizer import summarize_text
from qa_engine import chunk_text, embed_chunks, create_faiss_index, search_similar_chunks
from llm_answer import generate_answer
from fallback_extract import extract_email_and_phone

# Set up logging
logging.basicConfig(level=logging.INFO, filename="ai_assistant.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="AI Analyst Assistant", layout="wide")
st.title("🧠 AI Analyst Assistant")

# Sidebar input
st.sidebar.title("📁 File Upload")
file_type = st.sidebar.radio("Choose file type:", ["CSV", "PDF"])
uploaded_file = st.sidebar.file_uploader("Upload file", type=["csv", "pdf"])
use_llm = st.sidebar.checkbox("Use LLM to generate answers", value=True)

# ========== CSV MODULE ==========
if uploaded_file and file_type == "CSV":
    try:
        df = load_csv(uploaded_file)
        if df is not None:
            st.success("✅ CSV loaded successfully!")
            st.subheader("👀 Preview of Data")
            preview = get_preview(df)
            st.dataframe(preview["head"])
            st.dataframe(preview["tail"])

            st.subheader("📈 Data Summary")
            summary = get_data_summary(df)
            st.markdown(f"**Shape:** {summary['shape'][0]} rows × {summary['shape'][1]} columns")
            st.markdown(f"**Columns:** {', '.join(summary['columns'])}")
            st.subheader("📦 Column Types")
            st.json(summary["column_types"])
            st.subheader("🧩 Missing Values")
            st.json(summary["missing_values"])
            st.subheader("📊 Numeric Summary")
            st.json(summary["numeric_summary"])

            # Convert CSV to text for question-answering
            csv_text = df.to_string()
            chunks = chunk_text([{"page": 1, "text": csv_text}])
            embeddings, texts = embed_chunks(chunks)
            index = create_faiss_index(embeddings)
            st.success("✅ CSV indexed for question-answering.")

            question = st.text_input("💬 Ask a question about the CSV:")
            if question:
                context_chunks = search_similar_chunks(index, question, texts, top_k=5)
                combined_context = "\n\n".join(context_chunks)
                history_prompt = ""
                for q, a in st.session_state.chat_history[-3:]:
                    history_prompt += f"Q: {q}\nA: {a}\n"
                full_prompt = f"{history_prompt}\n{combined_context}"
                with st.spinner("🧠 Generating answer..."):
                    answer = generate_answer(question, full_prompt)
                st.session_state.chat_history.append((question, answer))
                st.markdown("### 💡 Answer:")
                st.success(answer)
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        logging.error(f"CSV processing error: {str(e)}")

# ========== PDF MODULE ==========
elif uploaded_file and file_type == "PDF":
    try:
        with st.spinner("📄 Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        if not pdf_text:
            st.error("❌ No text found in the PDF.")
        else:
            st.success("✅ PDF loaded successfully!")
            for page in pdf_text:
                st.markdown(f"### 📄 Page {page['page']}")
                st.text_area("📃 Extracted Text", value=page["text"], height=200)
                with st.spinner("📝 Generating summary..."):
                    summary = summarize_text(page["text"])
                st.success("📌 Summary:")
                st.write(summary)
                st.markdown("---")

            with st.spinner("🧠 Building FAISS index..."):
                chunks = chunk_text(pdf_text)
                embeddings, texts = embed_chunks(chunks)
                index = create_faiss_index(embeddings)
            st.success("✅ You can now ask questions about the PDF.")

            question = st.text_input("💬 Ask a question about the PDF:")
            if question:
                # Fallback for contact queries
                if "email" in question.lower() or "phone" in question.lower():
                    contact = extract_email_and_phone(pdf_text)
                    answer = f"Email: {contact['email']}\nPhone: {contact['phone']}"
                else:
                    with st.spinner("🔍 Searching relevant content..."):
                        context_chunks = search_similar_chunks(index, question, texts, top_k=5)
                        combined_context = "\n\n".join(context_chunks)
                    history_prompt = ""
                    for q, a in st.session_state.chat_history[-3:]:
                        history_prompt += f"Q: {q}\nA: {a}\n"
                    full_prompt = f"{history_prompt}\n{combined_context}"
                    with st.spinner("🧠 Generating answer..."):
                        answer = generate_answer(question, full_prompt)
                st.session_state.chat_history.append((question, answer))
                st.markdown("### 💡 Answer:")
                st.success(answer)

            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("### 📚 Chat History")
                for i, (q, a) in enumerate(st.session_state.chat_history):
                    st.markdown(f"**Q{i+1}:** {q}")
                    st.markdown(f"**A{i+1}:** {a}")

            if st.sidebar.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.success("Chat history cleared.")
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        logging.error(f"PDF processing error: {str(e)}")

else:
    st.info("⬅️ Upload a CSV or PDF file to begin.")