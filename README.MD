# Tibetan-English Neural Machine Translation

This repository contains code for neural machine translation between Tibetan and English. The project implements both a baseline model and an improved model with state-of-the-art techniques.

## New Features and Improvements

Our latest update includes significant improvements to the translation model:

- **Added adapter-based fine-tuning** using pre-trained mT5 model for improved efficiency and performance
- **Enhanced Tibetan-specific preprocessing** for better handling of Tibetan script
- **Improved training process** with cosine annealing learning rate scheduling
- **Multi-metric evaluation** using BLEU, METEOR, chrF, and TER metrics
- **Mixed precision training** for faster training and larger batch sizes
- **Advanced error analysis** including length-based evaluation and error pattern detection

## Project Structure

```
tibetan-english-translator/
├── data/               # Data preprocessing scripts and utilities
├── models/             # Model implementations
│   ├── baseline/       # Original baseline model
│   └── improved_translator.py  # New improved adapter-based model
├── scripts/            # Training and evaluation scripts
│   ├── train_improved_model.py  # Training script for the improved model
│   └── evaluate_model.py        # Comprehensive evaluation script
└── README.md
```

## Requirements

Install the required packages using:

```bash
pip install torch transformers datasets evaluate accelerate wandb
```

Additional requirements:
- `adapters` - For efficient adapter-based fine-tuning
- `tqdm` - For progress bars
- `sentencepiece` - For tokenization

## Using the Improved Model

### Training

To train the improved model with adapter-based fine-tuning:

```bash
python scripts/train_improved_model.py \
  --data_dir ./data \
  --output_dir ./outputs \
  --pretrained_model google/mt5-small \
  --num_epochs 10 \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --fp16
```

For more options, run:
```bash
python scripts/train_improved_model.py --help
```

### Evaluation

To evaluate the model:

```bash
python scripts/evaluate_model.py \
  --model_path ./outputs/best_adapter \
  --pretrained_model google/mt5-small \
  --source_file ./data/test.bo \
  --target_file ./data/test.en
```

## Technical Details

### Adapter-based Fine-tuning

The improved model uses adapter-based fine-tuning, which:
1. Keeps most of the pre-trained model parameters frozen
2. Only trains small adapter modules inserted into the Transformer layers
3. Dramatically reduces the number of trainable parameters (typically <5% of the full model)
4. Enables efficient training on limited computational resources

### Tibetan-specific Processing

The model implements special handling for Tibetan script:
- Unicode normalization to NFC form (recommended for Tibetan)
- Special handling of Tibetan-specific punctuation like shad (།)
- Proper syllable boundary detection

### Learning Rate Scheduling

We use Cosine Annealing with Warm Restarts for the learning rate schedule, which:
- Starts with a warmup period
- Follows a cosine decay pattern
- Periodically restarts the learning rate to help escape local minima

## Citation

If you use this code, please cite:

```
@misc{tibetan-english-translator,
  author = {Original Authors and Contributors},
  title = {Tibetan-English Neural Machine Translation},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Girim9912/tibetan-english-translator}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.