# DnD

## Getting Started

To launch the app, run:

```bash
streamlit run main.py
```

## Project Structure

```bash
.
├── database/               # Database for storing D&D data for RAG
│   ├── classes/             
│   ├── items/
│   ├── monsters/
│   ├── races/
│   ├── spell/              # D&D spells list
│   └── spell_content/      # Detailed D&D spell information
├── datasets/               # Datasets for training and testing
├── examples/               # Example code
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
└── README.md
```
