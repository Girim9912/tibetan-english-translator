import argparse
import logging
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from models.improved_translator import ImprovedTibetanEnglishTranslator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data(test_file_bo, test_file_en):
    """Load test data from files"""
    with open(test_file_bo, "r", encoding="utf-8") as f_bo, open(test_file_en, "r", encoding="utf-8") as f_en:
        source_texts = [line.strip() for line in f_bo.readlines()]
        target_texts = [line.strip() for line in f_en.readlines()]
    
    return source_texts, target_texts

def compute_comprehensive_metrics(predictions, references):
    """
    Compute multiple evaluation metrics for translation quality
    """
    # Load metrics
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    chrf = evaluate.load("chrf")
    ter = evaluate.load("ter")  # Translation Error Rate
    
    # Compute metrics
    bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
    
    # For METEOR, we need to handle lists of references
    meteor_refs = [[ref] for ref in references[0]]  # Assuming only 1 reference per prediction
    meteor_score = meteor.compute(predictions=predictions, references=meteor_refs)["meteor"]
    
    chrf_score = chrf.compute(predictions=predictions, references=references)["score"]
    ter_score = ter.compute(predictions=predictions, references=references)["score"]
    
    return {
        "bleu": bleu_score,
        "meteor": meteor_score,
        "chrf": chrf_score,
        "ter": ter_score
    }

def analyze_errors(source_texts, predictions, references):
    """
    Analyze common error patterns in translations
    """
    # This is a simple placeholder for error analysis
    # In a real implementation, you'd have more sophisticated analysis
    
    # Length difference analysis
    length_diffs = [len(pred.split()) - len(ref.split()) for pred, ref in zip(predictions, references[0])]
    avg_length_diff = sum(length_diffs) / len(length_diffs)
    
    # Check for untranslated words (keeping Tibetan script)
    untranslated_count = 0
    tibetan_script_range = range(0x0F00, 0x0FFF)  # Unicode range for Tibetan script
    
    for pred in predictions:
        for char in pred:
            if ord(char) in tibetan_script_range:
                untranslated_count += 1
                break
    
    return {
        "avg_length_diff": avg_length_diff,
        "untranslated_percentage": untranslated_count / len(predictions) * 100
    }

def evaluate_by_length(source_texts, predictions, references):
    """
    Evaluate translation quality by source sentence length
    """
    # Group sentences by length
    length_buckets = {
        "short (1-5 words)": [],
        "medium (6-15 words)": [],
        "long (16+ words)": []
    }
    
    bleu = evaluate.load("bleu")
    
    for i, source in enumerate(source_texts):
        words = source.split()
        length = len(words)
        
        if length <= 5:
            bucket = "short (1-5 words)"
        elif length <= 15:
            bucket = "medium (6-15 words)"
        else:
            bucket = "long (16+ words)"
            
        length_buckets[bucket].append(i)
    
    # Calculate BLEU for each length bucket
    results = {}
    for bucket, indices in length_buckets.items():
        if not indices:  # Skip empty buckets
            continue
            
        bucket_predictions = [predictions[i] for i in indices]
        bucket_references = [[references[0][i]] for i in indices]
        
        if bucket_predictions and bucket_references:
            score = bleu.compute(predictions=bucket_predictions, references=bucket_references)["bleu"]
            results[bucket] = {
                "bleu": score,
                "count": len(indices),
                "percentage": len(indices) / len(source_texts) * 100
            }
    
    return results

def main(args):
    """Main evaluation function"""
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    logger.info(f"Loading model from {args.model_path}")
    model = ImprovedTibetanEnglishTranslator(
        pretrained_model_name=args.pretrained_model,
        device=device
    )
    
    # Load adapter weights
    model.load_adapter(args.model_path)
    logger.info("Model loaded successfully")
    
    # Load test data
    source_texts, target_texts = load_test_data(args.source_file, args.target_file)
    logger.info(f"Loaded {len(source_texts)} test examples")
    
    # Generate translations
    logger.info("Generating translations...")
    batch_size = args.batch_size
    all_translations = []
    
    for i in tqdm(range(0, len(source_texts), batch_size)):
        batch_texts = source_texts[i:i+batch_size]
        translations = model.generate_translation(
            batch_texts,
            max_length=args.max_length,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty
        )
        all_translations.extend(translations)
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    references = [[target_texts]]  # Format expected by the metrics
    metrics = compute_comprehensive_metrics(all_translations, references)
    
    # Additional analyses
    logger.info("Analyzing translation patterns...")
    error_analysis = analyze_errors(source_texts, all_translations, references)
    length_analysis = evaluate_by_length(source_texts, all_translations, references)
    
    # Save results
    results = {
        "metrics": metrics,
        "error_analysis": error_analysis,
        "length_analysis": length_analysis
    }
    
    # Save translations
    translations_output = []
    for i, (source, target, pred) in enumerate(zip(source_texts, target_texts, all_translations)):
        translations_output.append({
            "id": i,
            "source": source,
            "reference": target,
            "prediction": pred
        })
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write results to files
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    with open(os.path.join(args.output_dir, "translations.json"), "w", encoding="utf-8") as f:
        json.dump(translations_output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logger.info("Evaluation Results:")
    logger.info(f"BLEU: {metrics['bleu']:.4f}")
    logger.info(f"METEOR: {metrics['meteor']:.4f}")
    logger.info(f"chrF: {metrics['chrf']:.4f}")
    logger.info(f"TER: {metrics['ter']:.4f}")
    logger.info(f"Average length difference: {error_analysis['avg_length_diff']:.2f} words")
    logger.info(f"Untranslated percentage: {error_analysis['untranslated_percentage']:.2f}%")
    
    logger.info("Results by sentence length:")
    for bucket, data in length_analysis.items():
        logger.info(f"  {bucket}: BLEU={data['bleu']:.4f}, {data['count']} examples ({data['percentage']:.1f}%)")
    
    logger.info(f"Full results saved to {args.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tibetan-English translation model")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model adapter directory")
    parser.add_argument("--pretrained_model", type=str, default="google/mt5-small", help="Base pretrained model")
    
    # Data arguments
    parser.add_argument("--source_file", type=str, required=True, help="Tibetan source file")
    parser.add_argument("--target_file", type=str, required=True, help="English reference file")
    
    # Generation arguments
    parser.add_argument("--max_length", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam size for beam search")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty for generation")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    
    # Other arguments
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)