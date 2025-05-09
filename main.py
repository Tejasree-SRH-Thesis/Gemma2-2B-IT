
import gradio as gr
import tempfile
import fitz  
import json  
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import re
import os

# os.environ["HF_TOKEN"] = "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"
# hf_token = os.getenv("HUGGINGFACE_TOKEN")
# if not hf_token:
#     raise EnvironmentError("Please set the HUGGINGFACE_TOKEN environment variable.")

MODEL_NAME = "google/gemma-2-2b-it"
DEVICE = "cpu"
hf_token = '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' #my personal hugging face token or please use your own token

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token, trust_remote_code=True)
    model_config = transformers.AutoConfig.from_pretrained(
        MODEL_NAME,
        use_auth_token=True
    )
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=model_config, device_map="cpu", token=hf_token, torch_dtype=torch.float32, trust_remote_code=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id)
    return generator



def extract_json(text):
    assistant_start = text.find("start_of_turn>model")
    if assistant_start == -1:
        return {"Error": "No assistant section found in output"}

    assistant_text = text[assistant_start:]
    assistant_text = re.sub(r"```(?:json)?|```", "", assistant_text).strip()

    start = assistant_text.find('{')
    if start == -1:
        return {"Error": "No opening '{' found in assistant section"}

    brace_count = 0
    for i in range(start, len(assistant_text)):
        if assistant_text[i] == '{':
            brace_count += 1
        elif assistant_text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = assistant_text[start:i+1]
                try:
                    return json.loads(json_str)
                except Exception as e:
                    return {"Error": f"JSON parse failed: {e}"}

    return {"Error": "No complete JSON block found"}

def build_prompt(text):
    instruction = f"""
You are an AI that extracts structured metadata from research papers.

Return ONLY valid JSON with the following structure and no extra text:

{{
  "Title": "Paper title",
  "Authors": ["Author 1", "Author 2"],
  "DOI": "DOI if available",
  "Keywords": ["Keyword1", "Keyword2"],
  "Abstract": "Abstract text",
  "Document Type": "Research Paper, Thesis, etc.",
  "Number of References": 10
}}

Here is the paper content:
{text[:2000]}
"""

    return (
        "<start_of_turn>user\n"
        + instruction.strip() +
        "\n<end_of_turn>\n<start_of_turn>model\n"
    )

def extract_metadata(generator, paper_text):
    prompt = build_prompt(paper_text)
    response = generator(prompt, max_new_tokens=1000, do_sample=False, temperature=0)
    raw_output = response[0]["generated_text"]
    return extract_json(raw_output)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text if text.strip() else "Error: No extractable text found in PDF."


def process_pdf(pdf_file):

    extracted_text = extract_text_from_pdf(pdf_file.name)
    if extracted_text.startswith("Error:"):
        return {"Error": "No extractable text found in the PDF."}
    metadata = extract_metadata(generator, extracted_text)
    return metadata

def main():
    global generator
    generator = load_model()
    iface = gr.Interface(          
        fn=process_pdf,
        inputs=gr.File(label="Upload PDF"),
        outputs="json",
        title="Metadata Extractor",
        description="Upload a PDF to extract metadata."
    ) #Gradio interface
    iface.launch() # Launch the interface

if __name__ == "__main__":
    main()



