import argparse
from .translator import BasicTranslator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Tibetan-English Translator')
    parser.add_argument('--text', '-t', required=True, help='Text to translate')
    parser.add_argument('--direction', '-d', choices=['tib2eng', 'eng2tib'], 
                        default='tib2eng', help='Translation direction')
    parser.add_argument('--dictionary', '-dict', default='src/data/raw/simple_dictionary.txt',
                        help='Path to dictionary file')
    
    args = parser.parse_args()
    
    # Initialize translator
    translator = BasicTranslator(dictionary_path=args.dictionary)
    
    # Perform translation
    if args.direction == 'tib2eng':
        translation = translator.translate_tibetan_to_english(args.text)
        print(f"Tibetan: {args.text}")
        print(f"English: {translation}")
    else:
        translation = translator.translate_english_to_tibetan(args.text)
        print(f"English: {args.text}")
        print(f"Tibetan: {translation}")

if __name__ == "__main__":
    main()