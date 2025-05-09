# Automated Metadata Extraction App

This project presents an automated system for extracting academic metadata from scientific documents using small-scale Large Language Models (LLMs) with fewer than 3 billion parameters. The solution leverages instruction-tuned models like google/gemma-2-2b-it and features an interactive Gradio web interface for real-time validation and user feedback. Our results show that small LLMs can achieve metadata extraction performance comparable to traditional NLP methods, while requiring significantly less computational resources. This project aims to improve the efficiency, scalability, and accessibility of academic metadata management across diverse publication formats.

## Getting Started
Follow these steps to get a copy of automated metadata extraction app and running on your local machine for development and testing purposes.

### Prerequisites
-**Anaconda**: Download and install Anaconda from the [official Anaconda website](https://www.anaconda.com/products/individual).
-**Hugging Face Account**: You need a [Hugging Face account](https://huggingface.co/) and a read access token.
  
## Setup Instructions

### Step 1. Clone the repository:
```bash
git clone https://github.com/your-username/gemma-gradio-app.git
cd gemma-gradio-app
```

### Step 2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### Step 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 4. Set your Hugging Face token:
Hugging face token can be found in your hugging face profile's setting -> Access Tokes -> Create new token (or) use your old token
** If you are using Linux/macOS:**
```bash
export HUGGINGFACE_TOKEN=your_token_here
```
** If you are using Windows CMD **
set HUGGINGFACE_TOKEN=your_token_here

### Step5 Run the App

```bash
python main.py
```
Once you run Step5, you can upload PDF files of scientific documents and test the metadata extraction.

### Notes:
**Using Local PDF Documents**:
  A few PDF documents available locally in `Data` folder that can be uploaded for testing purposes.
**Using My Existed Account**:
  You can directly use my hugging face token. The necessary details have been encapsulated within the code.
  
### Tech Stack:
- Python: Core programming language
- LangChain: Library for building language model applications
- Gemma Model : AI model for natural language understanding and generation

### Acknowledgments
Special thanks to Professor Binh Vu and SRH Heidelberg for supporting this project.
