# DnD

## Getting Started

To launch the app, run:

```bash
streamlit run main.py
```

## Project Structure

```bash
.
├── datasets/
│   ├── classes             
│   ├── items
│   ├── monsters
│   ├── races
│   ├── spell/              # D&D spells list
│   └── spell_content/      # Detailed D&D spell information
├── src/
│   ├── components          # Reusable parts    
│   │   └── sidebar.py     
│   ├── models
│   │   ├── best/           # LoRA weights for fine-tuned model
│   │   └── model_loader.py # Model loading and setup script
│   └── utils
│       ├── feedback_utils.py
│       └── mock.py
├── main.py                 # Streamlit app entry point
├── README.md
└── utils/
    ├── faiss_spell_index   # Vector store save directory
    ├── rag.ipynb           # RAG Code
    └── webscraping.ipynb   # Web scraping code for data collection
```
