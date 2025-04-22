from sacrebleu import corpus_bleu
from transformers import MarianMTModel, MarianTokenizer

def evaluate_model(model_path, test_data):
    """Evaluate a translation model on test data"""
    model = MarianMTModel.from_pretrained(model_path)
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    
    source_texts = [pair[0] for pair in test_data]
    reference_texts = [pair[1] for pair in test_data]
    
    # Generate translations
    translations = []
    for text in source_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**inputs)
        translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        translations.append(translation)
    
    # Calculate BLEU score
    bleu = corpus_bleu(translations, [reference_texts])
    
    return {
        "bleu": bleu.score,
        "translations": translations,
        "references": reference_texts
    }