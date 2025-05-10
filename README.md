# Dungeons and Dragons LLM as a Dungeons Master

A web-based, interactive Dungeons & Dragons experience with large language model (LLM). This project allows players to engage with a virtual Dungeon Master in a dynamic, story-driven environment—accessible.

## Getting Started

### Installation
Clone the repository:

```bash
git clone https://github.com/marshmeowllo/DnD.git
cd DnD
```

### Start the app:

To launch the app, run:

```bash
streamlit run main.py
```

## Features

- Interactive D&D gameplay with a language model as your Dungeon Master
- Retrieval-Augmented Generation (RAG).
- Multi-turn dialogue with persistent game state
- Fine-tunable with your own D&D data or campaign logs

## Project Structure

```bash
.
├── database               # Database for storing D&D data for RAG
├── datasets                # Datasets for training and testing
├── examples                # Example code
├── main.py
├── pages
├── src
│   ├── components          # Web components
│   ├── models              # Model loader
│   ├── state
│   ├── tools
│   └── utils
├── main.py                 # Streamlit app entry point
└── README.md
```
