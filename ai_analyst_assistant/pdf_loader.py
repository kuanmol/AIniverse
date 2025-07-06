# pdf_loader.py

import pdfplumber

def extract_text_from_pdf(file):
    """Extracts text from each page of the uploaded PDF."""
    all_text = []
    try:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    all_text.append({
                        "page": i + 1,
                        "text": text.strip()
                    })
        return all_text
    except Exception as e:
        print("Error while reading PDF:", e)
        return []
