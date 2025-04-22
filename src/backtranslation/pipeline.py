import torch
from transformers import MBartTokenizer, MBartForConditionalGeneration

class BackTranslationPipeline:
    def __init__(self, en_tib_model_path, tib_en_model_path):
        # Check if paths exist and load, otherwise use base mBART model
        try:
            print(f"Loading model from {en_tib_model_path}...")
            self.en_tib_model = MBartForConditionalGeneration.from_pretrained(en_tib_model_path)
            self.en_tib_tokenizer = MBartTokenizer.from_pretrained(en_tib_model_path)
        except Exception as e:
            print(f"Error loading English->Tibetan model: {e}")
            print("Using base mBART model instead")
            self.en_tib_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
            self.en_tib_tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")
        
        try:
            print(f"Loading model from {tib_en_model_path}...")
            self.tib_en_model = MBartForConditionalGeneration.from_pretrained(tib_en_model_path)
            self.tib_en_tokenizer = MBartTokenizer.from_pretrained(tib_en_model_path)
        except Exception as e:
            print(f"Error loading Tibetan->English model: {e}")
            print("Using base mBART model instead")
            self.tib_en_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
            self.tib_en_tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")
        
    def translate_batch(self, texts, model, tokenizer, max_length=128):
        # Helper function to translate a batch of texts
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        translated = model.generate(**inputs)
        return tokenizer.batch_decode(translated, skip_special_tokens=True)
        
    def generate_synthetic_data(self, mono_texts, model, tokenizer, batch_size=16):
        # Generate synthetic translations
        translations = []
        for i in range(0, len(mono_texts), batch_size):
            batch = mono_texts[i:i+batch_size]
            translations.extend(self.translate_batch(batch, model, tokenizer))
        
        # Return parallel data (original, translation)
        return list(zip(mono_texts, translations))