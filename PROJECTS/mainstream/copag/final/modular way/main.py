import os
import json
import gradio as gr
from document_processor import process_documents


def gradio_interface(files):
    if not files:
        return {"error": "No files uploaded"}, []
    result, commented_files = process_documents(files)
    return result, commented_files


if __name__ == "__main__":
    sample_files = [
        "adgm-ra-model-articles-private-company-limited-by-shares.docx",
        "adgm-ra-resolution-multiple-incorporate-shareholders-LTD-incorporation-v2.docx"
    ]
    result, commented_files = process_documents(sample_files)
    print("Process:", result["process"])
    print("Uploaded Documents:", result["documents_uploaded"])
    print("Uploaded Count:", result["documents_uploaded_count"])
    print("Required Count:", result["required_documents_count"])
    print("Missing Documents:", result["missing_documents"])
    print("Issues Found:", result["issues_found"])
    print("Message:", result["message"])
    print("Commented Files:", commented_files)
    print("Output saved to compliance_output.json")

    with gr.Blocks() as demo:
        gr.Markdown("# ADGM Corporate Agent Tool")
        gr.Markdown(
            "Upload .docx files to check compliance with ADGM regulations. Download commented files with flagged issues.")
        file_input = gr.File(file_count="multiple", file_types=[".docx"], label="Upload Documents")
        output_json = gr.JSON(label="Compliance Output")
        output_files = gr.File(label="Commented Documents")
        submit_button = gr.Button("Process Documents")
        submit_button.click(
            fn=gradio_interface,
            inputs=file_input,
            outputs=[output_json, output_files]
        )
    demo.launch()