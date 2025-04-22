from transformers import MBartTokenizer, MBartForConditionalGeneration
import torch

def main():
    print("Loading mBART model...")
    model_name = "facebook/mbart-large-50"
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    print("Setting up test sentences...")
    # Test with a simple English sentence
    en_text = "Hello, how are you today?"
    
    # For mBART, we need to set the source language
    tokenizer.src_lang = "en_XX"
    
    print("Tokenizing...")
    inputs = tokenizer(en_text, return_tensors="pt")
    
    print("Generating translation...")
    # Since we don't have a Tibetan-specific model, we'll use Chinese as a proxy
    tokenizer.tgt_lang = "zh_CN"  # Use Chinese as a proxy for demonstration
    outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang])
    
    print("Decoding translation...")
    translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Original: {en_text}")
    print(f"Translation: {translation}")
    
    # Now do back-translation
    print("\nPerforming back-translation...")
    tokenizer.src_lang = "zh_CN"  # Now Chinese is the source
    inputs = tokenizer(translation, return_tensors="pt")
    
    tokenizer.tgt_lang = "en_XX"  # English is the target
    outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[tokenizer.tgt_lang])
    
    back_translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"Back-translation: {back_translation}")

if __name__ == "__main__":
    main()