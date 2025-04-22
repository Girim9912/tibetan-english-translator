from sacrebleu import sentence_bleu
from sacremoses import MosesTokenizer
import re

def is_english(text):
    # Simple heuristic to detect English vs Tibetan
    # Check if text contains mostly Latin characters
    return bool(re.match(r'^[a-zA-Z\s\p{P}0-9]+$', text, re.UNICODE))

def filter_synthetic_data(original_texts, translated_texts, back_translated_texts, 
                          similarity_threshold=0.7):
    """Filter synthetic data based on quality metrics"""
    tokenizer = MosesTokenizer(lang='en')
    filtered_pairs = []
    
    for orig, trans, back_trans in zip(original_texts, translated_texts, back_translated_texts):
        # For English text, we can compute BLEU between original and back-translated
        if is_english(orig):
            orig_tokens = tokenizer.tokenize(orig.lower())
            back_tokens = tokenizer.tokenize(back_trans.lower())
            
            # Calculate sentence-level BLEU
            bleu_score = sentence_bleu(back_trans, [orig]).score / 100.0
            
            # Only keep examples with sufficient quality
            if bleu_score >= similarity_threshold:
                filtered_pairs.append((orig, trans))
        
        # For Tibetan, we might need other heuristics
        else:
            # Simplified: use length ratio as a proxy for quality
            length_ratio = len(trans) / max(1, len(orig))
            if 0.5 <= length_ratio <= 2.0:  # Reasonable length ratio
                filtered_pairs.append((orig, trans))
    
    return filtered_pairs