import os
import sys
import argparse

# Add the project directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import MBartForConditionalGeneration, MBartTokenizer
from src.backtranslation.pipeline import BackTranslationPipeline
from src.backtranslation.filtering import filter_synthetic_data
from src.data.data_collection import collect_english_texts, collect_tibetan_texts, save_monolingual_data
from src.models.training import train_translation_model
from src.models.evaluation import evaluate_model

def load_initial_parallel_data(data_path="data/parallel"):
    """Load existing parallel corpus"""
    parallel_data = []
    
    # Check if the file exists
    if os.path.exists(f"{data_path}/en-tib.txt"):
        with open(f"{data_path}/en-tib.txt", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    parallel_data.append((parts[0], parts[1]))
    
    # If no data, use a few examples
    if not parallel_data:
        parallel_data = [
            ("Hello.", "བཀྲ་ཤིས་བདེ་ལེགས།"),
            ("Thank you.", "ཐུགས་རྗེ་ཆེ།"),
            # Add more examples if available
        ]
    
    return parallel_data

def iterative_back_translation(mono_en_texts, mono_tib_texts, initial_parallel_data, 
                              num_iterations=3, models_dir="models"):
    """Run iterative back-translation process"""
    # Create directories if they don't exist
    os.makedirs(f"{models_dir}/en-tib", exist_ok=True)
    os.makedirs(f"{models_dir}/tib-en", exist_ok=True)
    
    # Train initial models if they don't exist
    en_tib_model_path = f"{models_dir}/en-tib/baseline"
    tib_en_model_path = f"{models_dir}/tib-en/baseline"
    
    if not os.path.exists(en_tib_model_path):
        print("Training initial English->Tibetan model...")
        train_translation_model(initial_parallel_data, en_tib_model_path, "en", "tib")
    
    if not os.path.exists(tib_en_model_path):
        print("Training initial Tibetan->English model...")
        # Swap source and target for the opposite direction
        swapped_data = [(pair[1], pair[0]) for pair in initial_parallel_data]
        train_translation_model(swapped_data, tib_en_model_path, "tib", "en")
    
    # Initialize pipeline
    pipeline = BackTranslationPipeline(en_tib_model_path, tib_en_model_path)
    
    # Keep track of our growing parallel corpus
    parallel_data = initial_parallel_data.copy()
    
    for iteration in range(num_iterations):
        print(f"Starting iteration {iteration+1}")
        
        # 1. Translate English monolingual data to Tibetan
        print("Translating English texts to Tibetan...")
        en_to_tib_synthetic = pipeline.generate_synthetic_data(
            mono_en_texts, 
            pipeline.en_tib_model, 
            pipeline.en_tib_tokenizer
        )
        
        # Save synthetic data
        os.makedirs(f"data/synthetic/iteration_{iteration+1}", exist_ok=True)
        with open(f"data/synthetic/iteration_{iteration+1}/en_to_tib.txt", "w", encoding="utf-8") as f:
            for en, tib in en_to_tib_synthetic:
                f.write(f"{en}\t{tib}\n")
        
        # 2. Train improved Tibetan->English model
        print("Training improved Tibetan->English model...")
        # Swap source and target for the correct direction
        synthetic_data = [(pair[1], pair[0]) for pair in en_to_tib_synthetic]
        
        # Combine with original parallel data (also swapped for correct direction)
        swapped_original = [(pair[1], pair[0]) for pair in parallel_data]
        train_tib_to_en_data = swapped_original + synthetic_data
        
        # Train new Tibetan->English model
        tib_en_model_path = f"{models_dir}/tib-en/iteration_{iteration+1}"
        train_translation_model(train_tib_to_en_data, tib_en_model_path, "tib", "en")
        
        # Update pipeline with new model
        pipeline.tib_en_model = MBartForConditionalGeneration.from_pretrained(tib_en_model_path)
        pipeline.tib_en_tokenizer = MBartTokenizer.from_pretrained(tib_en_model_path)
        
        # 3. Translate Tibetan monolingual data to English
        print("Translating Tibetan texts to English...")
        tib_to_en_synthetic = pipeline.generate_synthetic_data(
            mono_tib_texts, 
            pipeline.tib_en_model, 
            pipeline.tib_en_tokenizer
        )
        
        # Save synthetic data
        with open(f"data/synthetic/iteration_{iteration+1}/tib_to_en.txt", "w", encoding="utf-8") as f:
            for tib, en in tib_to_en_synthetic:
                f.write(f"{tib}\t{en}\n")
        
        # 4. Train improved English->Tibetan model
        print("Training improved English->Tibetan model...")
        train_en_to_tib_data = parallel_data + tib_to_en_synthetic
        
        # Train new English->Tibetan model
        en_tib_model_path = f"{models_dir}/en-tib/iteration_{iteration+1}"
        train_translation_model(train_en_to_tib_data, en_tib_model_path, "en", "tib")
        
        # Update pipeline with new model
        pipeline.en_tib_model = MBartForConditionalGeneration.from_pretrained(en_tib_model_path)
        pipeline.en_tib_tokenizer = MBartTokenizer.from_pretrained(en_tib_model_path)
        
        print(f"Completed iteration {iteration+1}")
    
    # Return paths to final models
    return {
        "en_tib_model": en_tib_model_path,
        "tib_en_model": tib_en_model_path
    }

def main():
    parser = argparse.ArgumentParser(description="Run iterative back-translation for Tibetan-English translation")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    args = parser.parse_args()
    
    # 1. Load initial parallel data
    print("Loading initial parallel data...")
    initial_parallel_data = load_initial_parallel_data(f"{args.data_dir}/parallel")
    print(f"Loaded {len(initial_parallel_data)} parallel sentences")
    
    # 2. Collect or load monolingual data
    mono_en_path = f"{args.data_dir}/monolingual/english/texts.txt"
    mono_tib_path = f"{args.data_dir}/monolingual/tibetan/texts.txt"
    
    if os.path.exists(mono_en_path) and os.path.exists(mono_tib_path):
        print("Loading existing monolingual data...")
        with open(mono_en_path, "r", encoding="utf-8") as f:
            english_texts = [line.strip() for line in f if line.strip()]
        
        with open(mono_tib_path, "r", encoding="utf-8") as f:
            tibetan_texts = [line.strip() for line in f if line.strip()]
    else:
        print("Collecting monolingual data...")
        english_texts = collect_english_texts(max_texts=1000)
        tibetan_texts = collect_tibetan_texts(max_texts=1000)
        save_monolingual_data(english_texts, tibetan_texts)
    
    print(f"Using {len(english_texts)} English texts and {len(tibetan_texts)} Tibetan texts")
    
    # 3. Run iterative back-translation
    print("Starting iterative back-translation...")
    model_paths = iterative_back_translation(
        english_texts[:500],  # Limit for faster iterations during development
        tibetan_texts[:500],
        initial_parallel_data,
        num_iterations=args.iterations,
        models_dir=args.models_dir
    )
    
    print("Iterative back-translation complete!")
    print(f"Final English->Tibetan model: {model_paths['en_tib_model']}")
    print(f"Final Tibetan->English model: {model_paths['tib_en_model']}")

if __name__ == "__main__":
    main()