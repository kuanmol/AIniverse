import re
import logging


def extract_email_and_phone(pages):
    """Extract email and phone numbers from text using regex."""
    full_text = " ".join([p["text"] for p in pages])
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'

    email = re.search(email_pattern, full_text)
    phone = re.search(phone_pattern, full_text)

    result = {
        "email": email.group(0) if email else "Not found",
        "phone": phone.group(0) if phone else "Not found"
    }
    logging.info(f"Extracted contact: {result}")
    return result