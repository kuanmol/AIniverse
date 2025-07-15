from transformers import pipeline

summarizer_pipeline = pipeline(
    "summarization",
    model="Falconsai/text_summarization",
    tokenizer="Falconsai/text_summarization"
)


def summarize_text(text, max_chunk_words=300):
    """Summarize long text by breaking it into safe-sized chunks."""
    if not text.strip():
        return "No text to summarize."

    summaries = []
    words = text.split()
    for i in range(0, len(words), max_chunk_words):
        chunk = " ".join(words[i:i + max_chunk_words])
        try:
            summary = summarizer_pipeline(
                chunk,
                max_length=100,
                min_length=25,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            summaries.append("[Error summarizing chunk]")

    return " ".join(summaries)
