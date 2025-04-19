class TibetanTokenizer:
    """A simple tokenizer for Tibetan text"""
    
    def __init__(self):
        # In Tibetan, the ་ (tsheg) character separates syllables
        self.tsheg_char = '་'
        # Sentence ending markers
        self.sentence_end_markers = ['།', '༎']
    
    def tokenize_syllables(self, text):
        """Split Tibetan text into syllables based on tsheg character"""
        # Replace sentence end markers with tsheg for consistent tokenization
        for marker in self.sentence_end_markers:
            text = text.replace(marker, self.tsheg_char)
        
        # Split by tsheg and filter out empty strings
        syllables = [s for s in text.split(self.tsheg_char) if s]
        return syllables
    
    def tokenize_words(self, text):
        """
        A very basic approach to tokenize Tibetan into words
        Note: Accurate Tibetan word segmentation requires more sophisticated approaches
        """
        # This is a simplified approach - proper Tibetan word segmentation
        # requires more advanced techniques
        syllables = self.tokenize_syllables(text)
        # For this simple version, we'll just return syllables as tokens
        # In a real system, you'd want to combine syllables into words
        return syllables

class EnglishTokenizer:
    """A simple tokenizer for English text"""
    
    def tokenize(self, text):
        """Split English text into words"""
        # This is a very basic tokenization - you'd want more sophisticated
        # approaches for a real system
        return text.lower().split()

# Example usage
def test_tokenizers():
    tibetan_text = "ཁྱེད་རང་ག་འདྲས་ཡོད། ང་བདེ་པོ་ཡིན།"
    english_text = "How are you? I am well."
    
    tib_tokenizer = TibetanTokenizer()
    eng_tokenizer = EnglishTokenizer()
    
    print("Tibetan syllables:", tib_tokenizer.tokenize_syllables(tibetan_text))
    print("English tokens:", eng_tokenizer.tokenize(english_text))

if __name__ == "__main__":
    test_tokenizers()
    