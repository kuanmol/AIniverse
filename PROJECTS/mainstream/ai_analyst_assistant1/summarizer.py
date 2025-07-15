from transformers import pipeline
import logging
import re

summarizer_pipeline = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    tokenizer="facebook/bart-large-cnn"
)


def summarize_text(text, max_chunk_words=150):  # Reduced for short pages
    """Summarize text, prioritizing substantive content over placeholders."""
    if not text.strip():
        return "No text to summarize."

    # Remove repetitive placeholder text
    text = re.sub(r'This sample PDF file is provided by Sample-Files\.com.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()

    summaries = []
    words = text.split()
    for i in range(0, len(words), max_chunk_words):
        chunk = " ".join(words[i:i + max_chunk_words])
        try:
            summary = summarizer_pipeline(
                chunk,
                max_length=100,
                min_length=20,
                do_sample=False,
                num_beams=4  # Improve quality with beam search
            )[0]["summary_text"]
            # Filter out placeholder-like summaries
            if "sample-files.com" not in summary.lower() and len(summary.split()) > 5:
                summaries.append(summary)
        except Exception as e:
            logging.warning(f"Summary failed for chunk: {str(e)}")
            summaries.append("Summary unavailable for this section.")

    if not summaries:
        return "No substantive content to summarize."
    return " ".join(summaries)