# Option 3: Zero-Shot Cross-Lingual Transfer for Natural Language Inference

A comprehensive NLP research project implementing zero-shot cross-lingual transfer for text classification, specifically Natural Language Inference (NLI). Train on English data only and evaluate on 5+ typologically diverse languages without any target-language training data.

## Project Overview

### Core Problem
Most NLP resources and models are English-centric, creating a significant gap for low-resource languages. This project explores how well multilingual models can transfer knowledge from English to other languages for complex tasks like NLI.

### Key Objectives
- **Train exclusively on English**: Fine-tune models using only English NLI data
- **Evaluate across languages**: Test on 6 languages (English, German, Chinese, Arabic, Russian, Hindi) without any training data in those languages
- **Compare transfer methods**: Direct transfer, adapter-based approaches, and partial fine-tuning strategies
- **Analyze linguistic factors**: Investigate how typological features (word order, morphology, script) affect transfer performance
- **Optimize for efficiency**: Use partial fine-tuning to reduce computational costs while maintaining performance

### Technical Challenges Addressed
- Script differences (Latin, Cyrillic, Arabic, Chinese characters)
- Vocabulary mismatch and out-of-vocabulary issues
- Linguistic phenomena (case systems, gender agreement, word order variations)
- English bias mitigation in multilingual representations

## Features

- **Multilingual NLI Classification**: Entailment, Neutral, Contradiction detection
- **Partial Fine-tuning**: Efficient training of ~7.86% of model parameters
- **Layer-wise Learning Rates**: Optimized training with different LRs for different model components
- **Interactive Demo**: Gradio web interface for real-time inference
- **Comprehensive Evaluation**: Per-language metrics and cross-lingual performance analysis
- **Checkpoint Management**: Incremental training with resumable checkpoints
- **Google Colab Integration**: Seamless cloud training with Drive storage

## Project Structure

```
nlp_project/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── app.py                       # Gradio web interface
├── configs/
│   └── config.yaml             # Hyperparameters and model configs
├── data/
│   ├── raw/                    # Original datasets (XNLI, MLDoc, etc.)
│   ├── processed/              # Tokenized and processed data
│   └── scripts/                # Data download and preprocessing scripts
├── models/
│   ├── baseline.py             # Direct transfer with mBERT/XLM-R
│   ├── advanced.py             # Adapter-based and advanced methods
│   └── utils.py                # Model loading, saving, metrics
├── src/
│   ├── preprocessing.py        # Text cleaning, tokenization, script handling
│   ├── training.py             # Training loops and English-only fine-tuning
│   ├── evaluation.py           # Per-language and aggregated metrics
│   ├── inference.py            # Cross-lingual prediction pipeline
│   └── error_analysis.py       # Detailed error analysis tools
├── notebooks/
│   └── Train Colab Notebook.ipynb  # End-to-end training and analysis
└── tests/                      # Unit tests for models and pipelines
```

## Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (recommended for GPU training)
- 8GB+ RAM (16GB+ recommended)
- Google account (for Colab training)

### Local Setup
```bash
# Clone the repository
git clone <repository-url>
cd nlp_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e .
```

### Google Colab Setup
Upload the `Train Colab Notebook.ipynb` to Google Colab and run the setup cells. The notebook includes:
- Automatic dependency installation
- Google Drive mounting for checkpoint storage
- GPU detection and device selection

## Model and Dataset Details

### Model Architecture
- **Primary Model**: XLM-RoBERTa-base (278M parameters, 100+ languages)
- **Task**: Sequence Classification for NLI (3 classes)
- **Fine-tuning Strategy**: Partial fine-tuning
  - Unfrozen components: Classifier head + Last 3 encoder layers
  - Trainable parameters: ~21.8M (7.86% of total)
  - Layer-wise learning rates:
    - Classifier: 1e-4
    - Layer 11 (top): 5e-5
    - Layers 9-10: 3e-5

### Dataset: XNLI
- **Task**: Natural Language Inference
- **Languages**: 15 languages total, we use 6 for evaluation
- **Classes**: entailment (0), neutral (1), contradiction (2)
- **Training**: English only (~392k examples)
- **Evaluation**: Zero-shot on all 6 languages
- **Format**: Premise-Hypothesis pairs with labels

**Evaluation Languages**:
- English (en) - High-resource baseline
- German (de) - Germanic, similar to English
- Chinese (zh) - Sino-Tibetan, logographic script
- Arabic (ar) - Afro-Asiatic, right-to-left script
- Russian (ru) - Slavic, Cyrillic script
- Hindi (hi) - Indo-Aryan, Devanagari script

## Usage

### Training Pipeline

#### Local Training
```bash
# Train the model
python src/training.py --config configs/config.yaml

# Evaluate on all languages
python src/evaluation.py --model models/checkpoints/
```

#### Google Colab Training
1. Open `notebooks/Train Colab Notebook.ipynb` in Colab
2. Mount Google Drive (automatic in notebook)
3. Run cells sequentially:
   - Setup & dependencies
   - Data loading & preprocessing
   - Model loading
   - Training (with checkpoints)
   - Evaluation & visualization

**Key Training Parameters**:
- Epochs: 3
- Batch size: 16
- Learning rate: Layer-wise (see above)
- Warmup ratio: 0.1
- Weight decay: 0.01
- Max sequence length: 128

### Checkpoint Management
The training saves checkpoints after each epoch for resumability:
```
checkpoints/
├── xnli_partial_ft_checkpoints/
│   ├── checkpoint_epoch_1.pt
│   ├── checkpoint_epoch_2.pt
│   └── checkpoint_epoch_3.pt
└── xnli_partial_ft/  # Final model
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer.json
```

### Resuming Training
```python
# In notebook or script
model, loss_history, val_acc_history = finetune_model(
    model, train_dataset, val_dataset,
    checkpoint_path="checkpoints/xnli_partial_ft_checkpoints",
    resume_epoch=2  # Resume from epoch 3
)
```

## Gradio Web Interface

### Launching the Demo
```bash
# Run the Gradio interface
python app.py
```

The interface will be available at `http://localhost:7860`

### Using the Interface
1. **Input**: Enter premise and hypothesis text in any supported language
2. **Model Selection**: Choose between different fine-tuned checkpoints
3. **Prediction**: Get real-time NLI predictions with confidence scores
4. **Batch Processing**: Upload CSV files for bulk predictions
5. **Visualization**: View attention weights and token importance

### Interface Features
- **Multilingual Support**: Input in any of the 6 evaluation languages
- **Confidence Scores**: Probability distributions for all 3 classes
- **Error Analysis**: Detailed explanations for predictions
- **Comparison Mode**: Compare predictions across different models
- **Export Results**: Download predictions as JSON/CSV

## Fine-tuning Pipeline Details

### 1. Data Preparation
- Load XNLI dataset using Hugging Face `datasets`
- Tokenize with XLM-RoBERTa tokenizer
- Handle multilingual text encoding
- Create DataLoader with random sampling

### 2. Model Setup
- Load pre-trained XLM-RoBERTa-base
- Set correct label mapping (entailment/neutral/contradiction)
- Freeze all parameters except classifier + last 3 layers
- Apply layer-wise learning rates for optimization

### 3. Training Loop
- AdamW optimizer with parameter groups
- Linear scheduler with warmup
- Gradient accumulation for stable training
- Progress tracking with tqdm
- Validation every epoch
- Checkpoint saving

### 4. Evaluation
- Per-language accuracy calculation
- Macro-averaged metrics
- Performance gap analysis from English
- Confusion matrix generation

### 5. Analysis
- Training curves visualization
- Cross-lingual performance plots
- Error pattern identification
- Linguistic feature correlation analysis

## Expected Results

### Performance Metrics
- **English (Training)**: ~85-90% accuracy
- **Similar Languages** (German): ~75-80% accuracy
- **Distant Languages** (Chinese, Arabic): ~65-75% accuracy
- **Average Transfer**: ~70-75% across all languages

### Key Insights
- Typological similarity correlates with transfer success
- Script differences create larger barriers than expected
- Partial fine-tuning maintains efficiency while preserving performance
- Layer-wise LRs significantly improve convergence

## Testing

Run the test suite:
```bash
pytest tests/
```

Test categories:
- Model loading and parameter freezing
- Tokenization correctness
- Training loop functionality
- Evaluation metrics accuracy
- Checkpoint saving/loading

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Use type hints for function signatures
- Maintain backward compatibility

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Hugging Face** for transformers and datasets libraries
- **Facebook AI** for XLM-RoBERTa model
- **XNLI Authors** for the multilingual NLI dataset
- **Gradio** for the web interface framework

## Contact

For questions or collaboration opportunities:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Note**: This project demonstrates advanced cross-lingual transfer techniques and serves as a foundation for multilingual NLP research. The code is optimized for both research and practical deployment scenarios.