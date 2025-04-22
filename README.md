# DnD

## Getting Started

To launch the app, run:

```bash
streamlit run main.py
```

## Project Structure

```bash
.
├── best/                  # LoRA weights for fine-tuned model
├── datasets/
│   ├── classes             
│   ├── items
│   ├── monsters
│   ├── races
│   ├── spell/             # D&D spells list
│   └── spell_content/     # Detailed D&D spell information
├── main.py                # Streamlit app entry point
├── model_loader.py        # Model loading and setup script
├── README.md
└── utils/
    ├── faiss_spell_index  # Vector store save directory
    ├── rag.ipynb          # RAG Code
    └── webscraping.ipynb  # Web scraping code for data collection
```
