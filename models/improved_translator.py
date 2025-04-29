import torch
import torch.nn as nn
from transformers import MT5ForConditionalGeneration, MT5Config
from transformers import AutoTokenizer
# from transformers.adapters import PfeifferConfig
# from adapters import PfeifferConfig
from adapters.configuration import PfeifferConfig
import logging

class ImprovedTibetanEnglishTranslator(nn.Module):
    """
    An improved Tibetan-English translator model that:
    1. Uses the mT5 pre-trained model as base
    2. Implements adapter-based fine-tuning for efficiency
    3. Has Tibetan-specific tokenization handling
    """
    
    def __init__(
        self, 
        pretrained_model_name="google/mt5-small", 
        adapter_name="tibetan_english_adapter",
        adapter_config=None,
        freeze_base_model=True,
        device=None
    ):
        super(ImprovedTibetanEnglishTranslator, self).__init__()
        
        # Set up device
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Load the pre-trained mT5 model
        logging.info(f"Loading pre-trained model: {pretrained_model_name}")
        self.model = MT5ForConditionalGeneration.from_pretrained(pretrained_model_name)
        
        # Load tokenizer with special handling for Tibetan script
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name,
            additional_special_tokens=["<bo>", "<en>"]  # Special tokens for language control
        )
        
        # Resize token embeddings to account for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Set up adapter
        if adapter_config is None:
            # Default adapter configuration
            adapter_config = PfeifferConfig(
                reduction_factor=16,  # Smaller reduction factor for better performance
                non_linearity="gelu",
            )
        
        # Add and activate the adapter
        self.model.add_adapter(adapter_name, config=adapter_config)
        self.model.train_adapter(adapter_name)
        
        # Freeze the base model parameters if specified
        if freeze_base_model:
            for param in self.model.parameters():
                param.requires_grad = False
                
        # Move the model to the appropriate device
        self.model.to(self.device)
        self.adapter_name = adapter_name
        
    def preprocess_tibetan_text(self, text):
        """
        Apply Tibetan-specific preprocessing:
        1. Normalize Unicode representations
        2. Handle syllable boundaries
        """
        # Basic normalization for Tibetan text
        # This would be expanded with more comprehensive Tibetan-specific processing
        
        # Normalize Unicode to NFC form which is recommended for Tibetan
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Add space after shad (Tibetan sentence delimiter: ། ) if not present
        text = text.replace('།', '། ')
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def tokenize(self, source_texts, target_texts=None, max_length=512):
        """
        Tokenize input texts with special handling for Tibetan
        """
        # Preprocess Tibetan text
        preprocessed_sources = [self.preprocess_tibetan_text(text) for text in source_texts]
        
        # Prepare inputs for the tokenizer
        model_inputs = self.tokenizer(
            preprocessed_sources,
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        )
        
        # If target texts are provided, tokenize them as well
        if target_texts:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    target_texts,
                    max_length=max_length,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                
            model_inputs["labels"] = labels
            
        return model_inputs
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass through the model
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            adapter_names=[self.adapter_name]
        )
        
        return outputs
    
    def generate_translation(
        self, 
        texts, 
        source_lang="bo", 
        target_lang="en", 
        max_length=100, 
        num_beams=5,
        length_penalty=1.0
    ):
        """
        Generate translations from source language to target language
        """
        # Add language tags if needed
        if source_lang == "bo":  # Tibetan
            preprocessed_texts = [self.preprocess_tibetan_text(text) for text in texts]
        else:
            preprocessed_texts = texts
            
        # Tokenize the input texts
        tokenized_inputs = self.tokenizer(
            preprocessed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Generate translations
        translated_ids = self.model.generate(
            input_ids=tokenized_inputs.input_ids,
            attention_mask=tokenized_inputs.attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
            adapter_names=[self.adapter_name]
        )
        
        # Decode the generated token IDs to text
        translations = self.tokenizer.batch_decode(
            translated_ids, 
            skip_special_tokens=True
        )
        
        return translations
    
    def save_adapter(self, output_dir):
        """Save only the adapter weights"""
        self.model.save_adapter(output_dir, self.adapter_name)
        # Save tokenizer for consistency
        self.tokenizer.save_pretrained(output_dir)
        
    def load_adapter(self, adapter_dir):
        """Load the adapter weights"""
        self.model.load_adapter(adapter_dir, adapter_name=self.adapter_name)
        # Activate the adapter
        self.model.set_active_adapters(self.adapter_name)