from transformers import MBartForConditionalGeneration, MBartConfig, MBartTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import numpy as np
import os
import torch

def prepare_dataset(parallel_data, tokenizer, max_length=128):
    """Convert parallel data to HuggingFace dataset with tokenization"""
    source_texts = [pair[0] for pair in parallel_data]
    target_texts = [pair[1] for pair in parallel_data]
    
    # Tokenize the inputs and targets
    model_inputs = tokenizer(source_texts, max_length=max_length, truncation=True, padding="max_length")
    
    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_texts, max_length=max_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    
    return Dataset.from_dict(model_inputs)

def train_translation_model(parallel_data, output_path, source_lang, target_lang, base_model=None):
    """Train a translation model on parallel data"""
    # Check if the model exists
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # For Tibetan, we'll use mBART which has better multilingual capabilities
    try:
        if base_model and os.path.exists(base_model):
            # Load from a previously trained checkpoint
            tokenizer = MBartTokenizer.from_pretrained(base_model)
            model = MBartForConditionalGeneration.from_pretrained(base_model)
        else:
            # Initialize from Facebook's mBART which supports many languages
            tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")
            model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
            
            # Configure for Tibetan if needed
            if target_lang == "tib" or source_lang == "tib":
                # Add Tibetan token to the tokenizer if not present
                if "tib" not in tokenizer.lang_code_to_id:
                    print("Adding Tibetan language code to tokenizer...")
                    # Get the next available token ID
                    new_lang_id = len(tokenizer.lang_code_to_id)
                    # Add Tibetan to the tokenizer
                    tokenizer.lang_code_to_id["tib"] = new_lang_id
                    tokenizer.id_to_lang_code[new_lang_id] = "tib"
                    # Add special tokens
                    special_tokens_dict = {'additional_special_tokens': ['<tib>']}
                    tokenizer.add_special_tokens(special_tokens_dict)
                    # Resize embeddings to account for new tokens
                    model.resize_token_embeddings(len(tokenizer))
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("Falling back to smaller mBART model...")
        # Fallback to smaller model
        tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-50")
        model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    
    # Prepare dataset
    print("Preparing dataset...")
    dataset = prepare_dataset(parallel_data, tokenizer)
    
    # Split into train/validation
    print("Splitting dataset...")
    dataset = dataset.train_test_split(test_size=0.1)
    
    # Define training arguments
    print("Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_path,
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=8,  # Reduced batch size for memory
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,  # More epochs for low-resource
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Use fp16 if GPU available
        push_to_hub=False,
        logging_dir=f"{output_path}/logs",
        logging_steps=10,
    )
    
    # Define trainer
    print("Setting up trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path