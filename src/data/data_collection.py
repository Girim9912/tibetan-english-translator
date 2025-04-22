import wikipediaapi
import requests
from bs4 import BeautifulSoup

def collect_english_texts(max_texts=1000):
    """Collect monolingual English texts from Wikipedia"""
    english_texts = []
    
    # Wikipedia API for English articles - add proper user agent
    wiki_en = wikipediaapi.Wikipedia(
        user_agent='tibetan-english-translator-project/1.0 (https://github.com/Girim9912/tibetan-english-translator)',
        language='en'
    )
    
    # Rest of the function remains the same
    topics = ['History', 'Science', 'Culture', 'Geography', 'Philosophy', 
              'Education', 'Technology', 'Art', 'Music', 'Literature']
    
    for topic in topics:
        page = wiki_en.page(topic)
        if page.exists():
            # Extract paragraphs and clean up
            paragraphs = page.text.split('\n\n')
            # Filter out very short paragraphs and references
            clean_paragraphs = [p for p in paragraphs if len(p.split()) > 10 and not p.startswith('==')]
            english_texts.extend(clean_paragraphs[:100])  # Limit to 100 paragraphs per topic
            
            if len(english_texts) > max_texts:
                english_texts = english_texts[:max_texts]
                break
    
    return english_texts

def collect_tibetan_texts(max_texts=1000):
    """Collect monolingual Tibetan texts from various sources"""
    tibetan_texts = []
    
    # Example sources (would need to implement specific scrapers for each)
    sources = [
        "https://www.tibettimes.net/",
        "https://www.rfa.org/tibetan/",
        # Add more sources
    ]
    
    # Simple example of scraping (would need appropriate permissions and more robust implementation)
    for source in sources:
        try:
            response = requests.get(source, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract paragraphs (this is simplified; actual implementation depends on site structure)
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    text = p.get_text().strip()
                    # Basic filtering for Tibetan text
                    if text and any(0x0F00 <= ord(c) <= 0x0FFF for c in text):
                        tibetan_texts.append(text)
                        
            if len(tibetan_texts) > max_texts:
                tibetan_texts = tibetan_texts[:max_texts]
                break
                
        except Exception as e:
            print(f"Error scraping {source}: {e}")
    
    # In a real implementation, we'd also read from local files if available
    
    return tibetan_texts

def save_monolingual_data(english_texts, tibetan_texts):
    """Save collected monolingual data to files"""
    import os
    
    # Ensure directories exist
    os.makedirs("data/monolingual/english", exist_ok=True)
    os.makedirs("data/monolingual/tibetan", exist_ok=True)
    
    # Save English texts
    with open("data/monolingual/english/texts.txt", "w", encoding="utf-8") as f:
        for text in english_texts:
            f.write(text + "\n")
    
    # Save Tibetan texts
    with open("data/monolingual/tibetan/texts.txt", "w", encoding="utf-8") as f:
        for text in tibetan_texts:
            f.write(text + "\n")