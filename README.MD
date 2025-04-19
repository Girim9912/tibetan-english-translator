# Tibetan-English Translator

A machine translation system for translating between Tibetan and English languages.

## Project Overview
This project aims to create a translation model that can convert text between Tibetan and English. The development follows a step-by-step approach, starting with basic dictionary-based translation and eventually implementing neural machine translation techniques.

## Project Structure
- `data/`: Dataset storage
- `src/`: Source code
- `notebooks/`: Jupyter notebooks for exploration and experimentation
- `models/`: Saved model files
- `docs/`: Documentation

## Setup Instructions
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/Mac)
4. Install dependencies: `pip install -r requirements.txt`

## Usage
To translate text using the command-line interface:

python -m src.translate_cli --text "ཁྱེད་རང་ག་འདྲས་ཡོད།" --direction tib2eng

python -m src.translate_cli --text "How are you?" --direction eng2tib


## Progress
- Repository created
- Project structure initialized
- Basic dictionary-based translator implemented
- Command-line interface for translation