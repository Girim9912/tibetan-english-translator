class TibetanEnglishDictionary:
    """A simple dictionary for Tibetan-English translation"""
    
    def __init__(self):
        # Initialize with some basic vocabulary
        self.tib_to_eng = {}
        self.eng_to_tib = {}
    
    def load_dictionary_from_file(self, file_path):
        """Load dictionary entries from a tab-separated file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            tibetan, english = parts
                            self.add_entry(tibetan, english)
            
            print(f"Loaded {len(self.tib_to_eng)} entries from {file_path}")
        except FileNotFoundError:
            print(f"Dictionary file not found: {file_path}")
    
    def add_entry(self, tibetan, english):
        """Add a dictionary entry"""
        self.tib_to_eng[tibetan] = english
        self.eng_to_tib[english] = tibetan
    
    def lookup_tibetan(self, tibetan_word):
        """Look up a Tibetan word"""
        return self.tib_to_eng.get(tibetan_word, None)
    
    def lookup_english(self, english_word):
        """Look up an English word"""
        return self.eng_to_tib.get(english_word, None)

# Test function
def test_dictionary():
    dictionary = TibetanEnglishDictionary()
    
    # Add some sample entries
    dictionary.add_entry("ཁྱེད་རང་", "you")
    dictionary.add_entry("ང་", "I")
    dictionary.add_entry("ཡིན", "am")
    
    # Test lookup
    print(f"'ང་' translates to: {dictionary.lookup_tibetan('ང་')}")
    print(f"'you' translates to: {dictionary.lookup_english('you')}")

if __name__ == "__main__":
    test_dictionary()