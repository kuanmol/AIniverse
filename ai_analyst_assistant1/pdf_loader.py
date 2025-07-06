import io
import logging
import re
import unicodedata
from pdfminer.high_level import extract_pages, extract_text
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image, ImageEnhance


def clean_text(text):
    """Clean extracted text to handle OCR artifacts, LaTeX symbols, and invalid Unicode."""
    if not text:
        return ""

    # Normalize Unicode to NFC
    text = unicodedata.normalize('NFC', text)

    # Remove surrogate characters (U+D800 to U+DFFF)
    text = re.sub(r'[\ud800-\udfff]', '', text)

    # Replace LaTeX/math symbols with placeholders
    text = re.sub(r'\$[^\$]+\$', ' [SYMBOL] ', text)

    # Remove non-printable/control characters
    text = re.sub(r'[^\x20-\x7E\u00A0-\uFFFF]', ' ', text)

    # Normalize table structures
    text = re.sub(r'\|\s*([^|]+)\s*\|', r'| \1 |', text)

    # Remove repetitive placeholder text
    text = re.sub(r'This sample PDF file is provided by Sample-Files\.com.*$', '', text, flags=re.MULTILINE)

    # Preserve list structures (e.g., "1. Introduction")
    text = re.sub(r'(\d+\.\s+)', r'\n\1', text)

    # Replace multiple spaces/tabs with a single space, preserve newlines for lists
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text).strip()

    # Encode to UTF-8, ignoring errors
    try:
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
    except Exception as e:
        logging.warning(f"UTF-8 encoding cleanup failed: {str(e)}")

    return text


def extract_text_from_pdf(file):
    """
    Extract text from a PDF using pdfminer.six, falling back to OCR if needed.
    Returns a list of dictionaries with page number and extracted text.
    """
    pages = []
    try:
        # Read file bytes
        file.seek(0)
        pdf_bytes = file.read()

        # Try pdfminer.six for text-based PDFs
        try:
            # Extract text per page to ensure correct page separation
            page_texts = []
            for page_layout in extract_pages(io.BytesIO(pdf_bytes)):
                text = extract_text(io.BytesIO(pdf_bytes), page_numbers=[len(page_texts)])
                text = clean_text(text)
                if text.strip():
                    page_texts.append(text)
                else:
                    page_texts.append("")
            for i, text in enumerate(page_texts, 1):
                if text.strip():
                    pages.append({"page": i, "text": text})
            if pages:
                logging.info("Text extracted successfully with pdfminer.six")
                return pages
        except Exception as e:
            logging.warning(f"pdfminer.six failed: {str(e)}. Falling back to OCR.")

        # Fallback to OCR with pdf2image and pytesseract
        images = convert_from_bytes(pdf_bytes)
        for i, image in enumerate(images, 1):
            try:
                # Enhance image: grayscale, increase contrast
                image = image.convert("L")
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)
                text = pytesseract.image_to_string(image, lang="eng+equ", config='--psm 6')
                text = clean_text(text)
                pages.append({"page": i, "text": text})
            except Exception as e:
                logging.error(f"OCR failed for page {i}: {str(e)}")
                pages.append({"page": i, "text": ""})
        logging.info(f"Text extracted via OCR for {len(pages)} pages")
        return pages

    except Exception as e:
        logging.error(f"PDF extraction failed: {str(e)}")
        return []