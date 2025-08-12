import os
from docx import Document
import docx2txt
from compliance_checker import check_red_flags, check_compliance, setup_rag
from config import define_checklists
import json
from docx_utils import add_docx_comment

def read_docx(file_path):
    try:
        if file_path.endswith('.docx'):
            doc = Document(file_path)
            text = [(i, para.text.strip()) for i, para in enumerate(doc.paragraphs) if para.text.strip()]
            return text, doc
        elif file_path.endswith('.doc'):
            text = docx2txt.process(file_path)
            text = [(i, line.strip()) for i, line in enumerate(text.split('\n')) if
                    line.strip() and not line.startswith('_') and 'EMBED' not in line]
            return text, None
        else:
            print(f"Unsupported file format: {file_path}")
            return [], None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return [], None

def classify_document(text, file_path):
    doc_types = [
        "Articles of Association",
        "Memorandum of Association",
        "Board Resolution Templates",
        "Shareholder Resolution Templates",
        "Incorporation Application Form",
        "UBO Declaration Form",
        "Register of Members and Directors",
        "Change of Registered Address Notice",
        "Licensing Application",
        "HR Contract"
    ]
    full_text = " ".join([t for _, t in text]).lower()
    if "resolution" in file_path.lower() and "shareholder" in full_text:
        print(f"Detected document type: Shareholder Resolution Templates in {full_text[:50]}...")
        return "Shareholder Resolution Templates"
    for doc_type in doc_types:
        if doc_type.lower() in full_text:
            print(f"Detected document type: {doc_type} in {full_text[:50]}...")
            return doc_type
    print(f"No document type detected in {full_text[:50]}...")
    return "Unknown"

def detect_process(documents):
    checklists = define_checklists()
    for process, required_docs in checklists.items():
        if any(doc in required_docs for doc in documents):
            print(f"Detected process: {process}")
            return process
    print("No process detected")
    return "Unknown"

def check_missing_documents(process, uploaded_docs):
    checklists = define_checklists()
    required_docs = checklists.get(process, [])
    missing = [doc for doc in required_docs if doc not in uploaded_docs]
    print(f"Missing documents: {missing}")
    return missing

def process_documents(file_paths):
    model, index, corpus = setup_rag()
    uploaded_docs = []
    all_issues = []
    commented_files = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        text, doc = read_docx(file_path)
        if text:
            doc_type = classify_document(text, file_path)
            uploaded_docs.append(doc_type)
            issues = check_red_flags(text, doc_type, doc)
            if model and index:
                compliance_issues = check_compliance(text, doc_type, model, index, corpus, doc)
                issues.extend(compliance_issues)
            all_issues.extend(issues)
            if doc:
                commented_path = f"commented_{os.path.basename(file_path)}"
                doc.save(commented_path)
                commented_files.append(commented_path)
    process = detect_process(uploaded_docs)
    missing_docs = check_missing_documents(process, uploaded_docs)
    required_count = len(define_checklists().get(process, []))
    uploaded_count = len(set(uploaded_docs))
    if process == "Unknown":
        message = "Unable to detect legal process. Please upload relevant ADGM documents."
    else:
        if not missing_docs:
            message = f"It appears that you’re trying to {process.lower()}. All required documents ({required_count}) have been uploaded."
        else:
            missing_str = ", ".join(missing_docs)
            message = (f"It appears that you’re trying to {process.lower()}. "
                       f"Based on our reference list, you have uploaded {uploaded_count} out of {required_count} required documents. "
                       f"The missing document(s): {missing_str}.")
    result = {
        "process": process,
        "documents_uploaded": list(set(uploaded_docs)),
        "documents_uploaded_count": uploaded_count,
        "required_documents_count": required_count,
        "missing_documents": missing_docs,
        "issues_found": all_issues,
        "message": message
    }
    with open("compliance_output.json", "w") as f:
        json.dump(result, f, indent=2)
    return result, commented_files