import re

def extract_email_and_phone(pages):
    full_text = " ".join([p['text'] for p in pages])
    email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', full_text)
    phone = re.search(r'\b\d{10}\b', full_text)
    return {
        "email": email.group(0) if email else "Not found",
        "phone": phone.group(0) if phone else "Not found"
    }
