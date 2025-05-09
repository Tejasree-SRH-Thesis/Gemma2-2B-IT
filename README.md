# Gemma Gradio App

This repository contains a Gradio interface to interact with the `google/gemma-2-2b-it` model.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/gemma-gradio-app.git
cd gemma-gradio-app
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your Hugging Face token:
You need a [Hugging Face account](https://huggingface.co/) and a read access token.
**Linux/macOS:**
```bash
export HUGGINGFACE_TOKEN=your_token_here
```
**Windows CMD**
set HUGGINGFACE_TOKEN=your_token_here

## Run the App

```bash
python main.py
```