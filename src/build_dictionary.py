from .data_loader import load_parallel_data
from .tokenizers import TibetanTokenizer, EnglishTokenizer

def build_dictionary_from_parallel_data(input_file, output_file):
    """Build a simple dictionary from parallel text data"""
    # Load the parallel data
    df = load_parallel_data(input_file)
    
    # Initialize tokenizers
    tib_tokenizer = TibetanTokenizer()
    eng_tokenizer = EnglishTokenizer()
    
    # Dictionary to store word pairs
    word_pairs = {}
    
    # Process each pair
    for _, row in df.iterrows():
        tibetan = row['tibetan']
        english = row['english']
        
        # Tokenize
        tib_syllables = tib_tokenizer.tokenize_syllables(tibetan)
        eng_words = eng_tokenizer.tokenize(english)
        
        # For this simple approach, we'll just map the first Tibetan syllable
        # to the first English word, second to second, etc.
        # This is very basic and won't work well for real translation,
        # but it's a starting point
        for i in range(min(len(tib_syllables), len(eng_words))):
            tib_word = tib_syllables[i]
            eng_word = eng_words[i]
            
            # Add to dictionary (overwriting if already exists)
            word_pairs[tib_word] = eng_word
    
    # Write the dictionary to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for tib, eng in word_pairs.items():
            f.write(f"{tib}\t{eng}\n")
    
    print(f"Built dictionary with {len(word_pairs)} entries and saved to {output_file}")

if __name__ == "__main__":
    input_file = 'data/raw/tibetan_english_sample.txt'
    output_file = 'data/processed/simple_dictionary.txt'
    build_dictionary_from_parallel_data(input_file, output_file)