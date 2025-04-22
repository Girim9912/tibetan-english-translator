import os
import sys
import argparse
import json

# Add the project directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.evaluation import evaluate_model
from src.backtranslation.pipeline import BackTranslationPipeline

def load_test_data(test_file):
    """Load test data from file"""
    test_data = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                test_data.append((parts[0], parts[1]))
    return test_data

def main():
    parser = argparse.ArgumentParser(description="Evaluate translation models")
    parser.add_argument("--en-tib-model", type=str, required=True, help="Path to English->Tibetan model")
    parser.add_argument("--tib-en-model", type=str, required=True, help="Path to Tibetan->English model")
    parser.add_argument("--test-file", type=str, required=True, help="Path to test data file")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    args = parser.parse_args()
    
    # Load test data
    test_data = load_test_data(args.test_file)
    print(f"Loaded {len(test_data)} test examples")
    
    # Evaluate English->Tibetan model
    print("Evaluating English->Tibetan model...")
    en_tib_results = evaluate_model(args.en_tib_model, test_data)
    
    # For Tibetan->English, we need to swap source and target
    swapped_test_data = [(pair[1], pair[0]) for pair in test_data]
    
    # Evaluate Tibetan->English model
    print("Evaluating Tibetan->English model...")
    tib_en_results = evaluate_model(args.tib_en_model, swapped_test_data)
    
    # Save results
    results = {
        "en_tib": {
            "bleu": en_tib_results["bleu"],
            "examples": [
                {"source": s, "reference": r, "translation": t} 
                for s, r, t in zip(
                    test_data[:10],  # Limit to first 10 examples for readability
                    en_tib_results["references"][:10],
                    en_tib_results["translations"][:10]
                )
            ]
        },
        "tib_en": {
            "bleu": tib_en_results["bleu"],
            "examples": [
                {"source": s, "reference": r, "translation": t} 
                for s, r, t in zip(
                    swapped_test_data[:10],
                    tib_en_results["references"][:10],
                    tib_en_results["translations"][:10]
                )
            ]
        }
    }
    
    # Save to file
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {args.output}")
    print(f"English->Tibetan BLEU: {en_tib_results['bleu']:.2f}")
    print(f"Tibetan->English BLEU: {tib_en_results['bleu']:.2f}")

if __name__ == "__main__":
    main