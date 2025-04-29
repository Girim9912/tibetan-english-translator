import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from accelerate import Accelerator
import evaluate
import wandb

from improved_translator import ImprovedTibetanEnglishTranslator
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    """Dataset for machine translation"""
    
    def __init__(self, source_texts, target_texts):
        self.source_texts = source_texts
        self.target_texts = target_texts
        
    def __len__(self):
        return len(self.source_texts)
        
    def __getitem__(self, idx):
        return {
            "source_text": self.source_texts[idx],
            "target_text": self.target_texts[idx]
        }


def load_data(data_dir=None, dataset_name=None, split_ratio=0.9):
    """
    Load translation data either from local files or from a Hugging Face dataset
    """
    if dataset_name:
        # Load from Hugging Face datasets
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        # Assuming the dataset has train and validation splits
        # and columns named 'bo' for Tibetan and 'en' for English
        train_bo = dataset["train"]["bo"]
        train_en = dataset["train"]["en"]
        
        if "validation" in dataset:
            val_bo = dataset["validation"]["bo"]
            val_en = dataset["validation"]["en"]
        else:
            # Create validation split from training data
            train_size = len(train_bo)
            val_size = int(train_size * (1 - split_ratio))
            
            indices = np.random.permutation(train_size)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            val_bo = [train_bo[i] for i in val_indices]
            val_en = [train_en[i] for i in val_indices]
            train_bo = [train_bo[i] for i in train_indices]
            train_en = [train_en[i] for i in train_indices]
            
    else:
        # Load from local files
        if not data_dir:
            raise ValueError("Either data_dir or dataset_name must be provided")
            
        bo_file = os.path.join(data_dir, "train.bo")
        en_file = os.path.join(data_dir, "train.en")
        
        with open(bo_file, "r", encoding="utf-8") as f_bo, open(en_file, "r", encoding="utf-8") as f_en:
            bo_lines = f_bo.readlines()
            en_lines = f_en.readlines()
            
        # Clean the lines
        bo_lines = [line.strip() for line in bo_lines]
        en_lines = [line.strip() for line in en_lines]
        
        # Create train/val split
        data_size = len(bo_lines)
        indices = np.random.permutation(data_size)
        train_size = int(data_size * split_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_bo = [bo_lines[i] for i in train_indices]
        train_en = [en_lines[i] for i in train_indices]
        val_bo = [bo_lines[i] for i in val_indices]
        val_en = [en_lines[i] for i in val_indices]
    
    logger.info(f"Training data: {len(train_bo)} examples")
    logger.info(f"Validation data: {len(val_bo)} examples")
    
    return train_bo, train_en, val_bo, val_en


def compute_metrics(predictions, references):
    """
    Compute evaluation metrics for translation quality
    """
    bleu = evaluate.load("bleu")
    bleu_score = bleu.compute(predictions=predictions, references=references)["bleu"]
    
    # You can add more metrics here (METEOR, chrF, etc.)
    
    return {
        "bleu": bleu_score
    }


def train(args):
    """
    Main training function for the improved translator model
    """
    # Initialize accelerator for mixed precision and distributed training
    accelerator = Accelerator(mixed_precision='fp16' if args.fp16 else 'no')
    
    # Set up wandb for experiment tracking
    if args.use_wandb:
        wandb.init(project="tibetan-english-translator", name=args.run_name)
        
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    train_bo, train_en, val_bo, val_en = load_data(
        data_dir=args.data_dir, 
        dataset_name=args.dataset_name,
        split_ratio=args.train_ratio
    )
    
    # Create datasets
    train_dataset = TranslationDataset(train_bo, train_en)
    val_dataset = TranslationDataset(val_bo, val_en)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize the model
    model = ImprovedTibetanEnglishTranslator(
        pretrained_model_name=args.pretrained_model,
        freeze_base_model=not args.train_full_model
    )
    
    # Set up optimizer with weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        eps=args.adam_epsilon
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.scheduler_t0,
        T_mult=args.scheduler_t_mult,
        eta_min=args.min_learning_rate
    )
    
    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Training loop
    global_step = 0
    best_bleu = 0.0
    
    logger.info("Starting training...")
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            source_texts = batch["source_text"]
            target_texts = batch["target_text"]
            
            # Tokenize batch
            inputs = model.tokenize(source_texts, target_texts, max_length=args.max_length)
            inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            
            loss = outputs.loss
            
            # Backward pass
            accelerator.backward(loss)
            
            if args.clip_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            total_loss += loss.item()
            
            # Logging
            if batch_idx % args.log_interval == 0:
                logger.info(
                    f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
                
                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/global_step": global_step
                    })
            
            # Evaluation
            if global_step % args.eval_steps == 0:
                bleu_score = evaluate_model(model, val_loader, accelerator)
                logger.info(f"Validation BLEU: {bleu_score:.4f}")
                
                if args.use_wandb:
                    wandb.log({
                        "eval/bleu": bleu_score,
                        "eval/global_step": global_step
                    })
                
                # Save best model
                if bleu_score > best_bleu:
                    best_bleu = bleu_score
                    logger.info(f"New best BLEU score: {best_bleu:.4f}. Saving model...")
                    
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)
                    
                    # Save only the adapter parts
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_adapter(os.path.join(args.output_dir, "best_adapter"))
                    
                    # Also save full model checkpoint if requested
                    if args.save_full_model:
                        accelerator.save_state(os.path.join(args.output_dir, "best_checkpoint"))
        
        # Log epoch statistics
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
                
    logger.info("Training completed!")
    logger.info(f"Best validation BLEU score: {best_bleu:.4f}")
    
    # Final evaluation on test set if available
    if args.test_data_dir:
        logger.info("Evaluating on test set...")
        # Load test data
        with open(os.path.join(args.test_data_dir, "test.bo"), "r", encoding="utf-8") as f_bo, \
             open(os.path.join(args.test_data_dir, "test.en"), "r", encoding="utf-8") as f_en:
            test_bo = [line.strip() for line in f_bo.readlines()]
            test_en = [line.strip() for line in f_en.readlines()]
        
        test_dataset = TranslationDataset(test_bo, test_en)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        test_loader = accelerator.prepare(test_loader)
        
        test_bleu = evaluate_model(model, test_loader, accelerator)
        logger.info(f"Test BLEU score: {test_bleu:.4f}")
        
        if args.use_wandb:
            wandb.log({"test/bleu": test_bleu})
    
    # Clean up
    if args.use_wandb:
        wandb.finish()


def evaluate_model(model, data_loader, accelerator):
    """
    Evaluate the model on the provided data loader
    """
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in data_loader:
            source_texts = batch["source_text"]
            target_texts = batch["target_text"]
            
            # Move batch to appropriate device
            source_texts = accelerator.pad_across_processes(source_texts)
            
            # Generate translations
            translations = model.generate_translation(source_texts)
            
            # Gather predictions and references from all processes
            translations = accelerator.gather_for_metrics(translations)
            target_texts = accelerator.gather_for_metrics(target_texts)
            
            all_predictions.extend(translations)
            all_references.extend([[text] for text in target_texts])  # BLEU expects list of references
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_references)
    
    model.train()
    return metrics["bleu"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train an improved Tibetan-English translator")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing source data files")
    parser.add_argument("--dataset_name", type=str, default=None, help="HuggingFace dataset name")
    parser.add_argument("--test_data_dir", type=str, default=None, help="Directory containing test data files")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train/validation split ratio")
    
    # Model arguments
    parser.add_argument("--pretrained_model", type=str, default="google/mt5-small", 
                       help="Pretrained model name or path")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--train_full_model", action="store_true", help="Train full model instead of just adapters")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for model")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient norm clipping")
    parser.add_argument("--scheduler_t0", type=int, default=1, help="T_0 parameter for cosine scheduler")
    parser.add_argument("--scheduler_t_mult", type=int, default=2, help="T_mult parameter for cosine scheduler")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Logging and saving
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval in steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation interval in steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the run for tracking")
    parser.add_argument("--save_full_model", action="store_true", help="Save full model checkpoints")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)