from .tokenizers import TibetanTokenizer, EnglishTokenizer
from .dictionary import TibetanEnglishDictionary

class BasicTranslator:
    """A simple dictionary-based translator for Tibetan-English"""
    
    def __init__(self, dictionary_path=None):
        self.tibetan_tokenizer = TibetanTokenizer()
        self.english_tokenizer = EnglishTokenizer()
        self.dictionary = TibetanEnglishDictionary()
        
        if dictionary_path:
            self.dictionary.load_dictionary_from_file(dictionary_path)
    
    def translate_tibetan_to_english(self, tibetan_text):
        """Translate Tibetan text to English using dictionary lookup"""
        # Tokenize Tibetan text into syllables
        syllables = self.tibetan_tokenizer.tokenize_syllables(tibetan_text)
        
        # Try to translate each token
        translations = []
        for syllable in syllables:
            english = self.dictionary.lookup_tibetan(syllable)
            if english:
                translations.append(english)
            else:
                # If not found in dictionary, keep the original syllable
                translations.append(f"[{syllable}]")
        
        # Join the translations
        return " ".join(translations)
    
    def translate_english_to_tibetan(self, english_text):
        """Translate English text to Tibetan using dictionary lookup"""
        # Tokenize English text
        tokens = self.english_tokenizer.tokenize(english_text)
        
        # Try to translate each token
        translations = []
        for token in tokens:
            tibetan = self.dictionary.lookup_english(token)
            if tibetan:
                translations.append(tibetan)
            else:
                # If not found in dictionary, keep the original token in brackets
                translations.append(f"[{token}]")
        
        # Join the translations (no spaces in Tibetan)
        return "".join(translations)

# Test function
def test_translator():
    # Create a translator with some sample entries
    translator = BasicTranslator()
    translator.dictionary.add_entry("ཁྱེད་རང་", "you")
    translator.dictionary.add_entry("ག་འདྲས་", "how")
    translator.dictionary.add_entry("ཡོད", "are")
    translator.dictionary.add_entry("ང་", "I")
    translator.dictionary.add_entry("བདེ་པོ་", "well")
    translator.dictionary.add_entry("ཡིན", "am")
    
    # Test translation
    tibetan_text = "ཁྱེད་རང་ག་འདྲས་ཡོད།"
    english_text = "I am well"
    
    english_translation = translator.translate_tibetan_to_english(tibetan_text)
    tibetan_translation = translator.translate_english_to_tibetan(english_text)
    
    print(f"Tibetan: {tibetan_text}")
    print(f"English translation: {english_translation}")
    print()
    print(f"English: {english_text}")
    print(f"Tibetan translation: {tibetan_translation}")

if __name__ == "__main__":
    test_translator()