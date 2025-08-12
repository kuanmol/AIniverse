import re
import numpy as np
from ollama import Client
from sentence_transformers import SentenceTransformer
import faiss
from config import create_adgm_corpus
from docx_utils import add_docx_comment

def setup_rag():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        corpus = create_adgm_corpus()
        embeddings = model.encode(corpus)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return model, index, corpus
    except Exception as e:
        print(f"RAG setup error: {e}. Using regex checks only.")
        return None, None, create_adgm_corpus()

def check_red_flags(text, doc_type, doc):
    issues = []
    current_section = "Unknown"
    for i, line in text:
        if re.match(r"(PART \d+[A-Z]?|Clause \d+\.\d+|Section \d+\.\d+|Article \d+)", line, re.IGNORECASE):
            current_section = line.strip()

    for para_index, line in text:
        if re.search(r"UAE Federal Courts", line, re.IGNORECASE):
            issue = {
                "document": doc_type,
                "section": current_section,
                "issue": "Jurisdiction clause does not specify ADGM",
                "severity": "High",
                "suggestion": "Update jurisdiction to ADGM Courts."
            }
            issues.append(issue)
            if doc:
                add_docx_comment(doc, para_index, issue["suggestion"])
            break

    if doc_type in ["Articles of Association", "Memorandum of Association"]:
        governing_law_found = False
        for para_index, line in text:
            if re.search(r"governing law|ADGM regulations", line, re.IGNORECASE):
                governing_law_found = True
                break
        if not governing_law_found:
            issue = {
                "document": doc_type,
                "section": current_section,
                "issue": "No governing law clause found",
                "severity": "Medium",
                "suggestion": "Add clause specifying ADGM regulations as governing law."
            }
            issues.append(issue)
            if doc:
                add_docx_comment(doc, text[0][0], issue["suggestion"])

    if doc_type == "Shareholder Resolution Templates":
        signature_found = any("Signature of Incorporating Shareholders" in line for _, line in text)
        if not signature_found or not any("Shareholder’s Signature" in line for _, line in text):
            issue = {
                "document": doc_type,
                "section": current_section,
                "issue": "Missing or incomplete signature section for shareholders",
                "severity": "High",
                "suggestion": "Add a complete signature block for all incorporating shareholders as per ADGM Companies Regulations 2020, Part 12."
            }
            issues.append(issue)
            if doc:
                add_docx_comment(doc, text[-1][0], issue["suggestion"])

    try:
        client = Client()
        full_text = " ".join([t for _, t in text])[:500]
        prompt = (
            f"Analyze the following legal document text for ambiguous or non-binding language "
            f"(e.g., use of 'may' instead of 'shall'). Identify the specific sentence or phrase, "
            f"its approximate location (e.g., paragraph number), and suggest a fix. "
            f"If none, say 'No ambiguous language detected.' Text: {full_text}"
        )
        response = client.generate(model="llama3.1:8b", prompt=prompt)
        llm_response = response["response"]
        print(f"LLM response for {doc_type}: {llm_response}")
        if "ambiguous" in llm_response.lower() or "non-binding" in llm_response.lower():
            para_match = re.search(r"paragraph (\d+)", llm_response, re.IGNORECASE)
            para_index = int(para_match.group(1)) if para_match and int(para_match.group(1)) < len(text) else text[0][0]
            issue = {
                "document": doc_type,
                "section": current_section,
                "issue": "Ambiguous or non-binding language detected",
                "severity": "Medium",
                "suggestion": llm_response if len(llm_response) < 500 else llm_response[:500] + "..."
            }
            issues.append(issue)
            if doc:
                add_docx_comment(doc, para_index, issue["suggestion"])
    except Exception as e:
        print(f"LLM error for {doc_type}: {e}. Skipping LLM checks.")
        issue = {
            "document": doc_type,
            "section": current_section,
            "issue": "LLM check failed due to connection error",
            "severity": "Low",
            "suggestion": "Ensure Ollama is running and retry."
        }
        issues.append(issue)
        if doc:
            add_docx_comment(doc, text[0][0], issue["suggestion"])

    return issues

def check_compliance(text, doc_type, model, index, corpus, doc):
    issues = []
    full_text = " ".join([t for _, t in text])[:500]
    current_section = "Unknown"
    for i, line in text:
        if re.match(r"(PART \d+[A-Z]?|Clause \d+\.\d+|Section \d+\.\d+|Article \d+)", line, re.IGNORECASE):
            current_section = line.strip()
    try:
        client = Client()
        if model and index:
            doc_embedding = model.encode([full_text])[0]
            _, indices = index.search(np.array([doc_embedding]), k=5)  # Increased k to 5
            relevant_regs = [corpus[i] for i in indices[0]]
        else:
            relevant_regs = corpus  # Fallback to entire corpus
            print(f"RAG unavailable for {doc_type}. Using full corpus for compliance check.")
        prompt = (
            f"Check the following document text for compliance with these ADGM regulations: {relevant_regs}. "
            f"For each non-compliance issue, specify: 1) the exact sentence or phrase, 2) its paragraph number, "
            f"3) the specific regulation violated from the provided list, and 4) a detailed fix. "
            f"If none, state 'No non-compliance issues detected.' Text: {full_text}"
        )
        response = client.generate(model="llama3.1:8b", prompt=prompt)
        llm_response = response["response"]
        print(f"RAG LLM response for {doc_type}: {llm_response}")
        if "non-compliance" in llm_response.lower():
            issue_lines = llm_response.split('\n')
            for issue_line in issue_lines:
                if "non-compliance" in issue_line.lower():
                    para_match = re.search(r"paragraph (\d+)", issue_line, re.IGNORECASE)
                    if para_match and int(para_match.group(1)) < len(text):
                        para_index = int(para_match.group(1))
                        suggestion = issue_line if len(issue_line) < 500 else issue_line[:500] + "..."
                    else:
                        phrase_match = re.search(r"['\"]([^'\"]+)['\"]", issue_line)
                        phrase = phrase_match.group(1) if phrase_match else None
                        para_index = text[0][0]
                        suggestion = issue_line if len(issue_line) < 500 else issue_line[:500] + "..."
                        if phrase:
                            for i, line in text:
                                if phrase.lower() in line.lower():
                                    para_index = i
                                    break
                    issue = {
                        "document": doc_type,
                        "section": current_section,
                        "issue": "Non-compliance with ADGM regulations",
                        "severity": "High",
                        "suggestion": suggestion
                    }
                    issues.append(issue)
                    if doc:
                        add_docx_comment(doc, para_index, issue["suggestion"])
    except Exception as e:
        print(f"RAG error for {doc_type}: {e}. Skipping RAG checks.")
        issue = {
            "document": doc_type,
            "section": current_section,
            "issue": "RAG check failed due to LLM connection error",
            "severity": "Low",
            "suggestion": "Ensure Ollama is running and retry."
        }
        issues.append(issue)
        if doc:
            add_docx_comment(doc, text[0][0], issue["suggestion"])
    return issues