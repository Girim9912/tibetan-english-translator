import os
import requests
import zipfile
import io

def create_data_directories():
    """Create necessary directories for data storage"""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    print("Created data directories")

def download_sample_data():
    """Download a sample Tibetan-English dataset"""
    # For this example, we'll use a small sample from OPUS (if available)
    # In reality, you might need to manually collect and prepare data
    
    print("Searching for Tibetan-English parallel text samples...")
    
    # Sample Tibetan phrases with English translations
    sample_data = [
        ("ཁྱེད་རང་ག་འདྲས་ཡོད།", "How are you?"),
        ("ང་བདེ་པོ་ཡིན།", "I am well."),
        ("ཐུགས་རྗེ་ཆེ།", "Thank you."),
        ("དགོངས་དག", "Sorry/Excuse me."),
        ("བཀྲ་ཤིས་བདེ་ལེགས།", "Hello/Greetings."),
        ("ཞོགས་པ་བདེ་ལེགས།", "Good morning."),
        ("མཚན་མོ་བདེ་ལེགས།", "Good night."),
        ("ཁྱེད་རང་གི་མིང་ལ་ག་རེ་ཞུ་གི་ཡོད།", "What is your name?"),
        ("ངའི་མིང་ལ་_ཟེར་གྱི་ཡོད།", "My name is _."),
        ("དགའ་བསུ་ཞུ།", "Welcome.")
    ]
    
    # Save the sample data
    with open('data/raw/tibetan_english_sample.txt', 'w', encoding='utf-8') as f:
        for tibetan, english in sample_data:
            f.write(f"{tibetan}\t{english}\n")
    
    print(f"Saved {len(sample_data)} sample phrase pairs to data/raw/tibetan_english_sample.txt")

if __name__ == "__main__":
    create_data_directories()
    download_sample_data()