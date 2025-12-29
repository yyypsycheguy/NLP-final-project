3. Zero-Shot Cross-Lingual Transfer
Context: Most NLP resources exist for English. Zero-shot cross-lingual transfer trains models on English data and tests on low-resource languages without any training data. Success depends on multilingual representations and typological similarity.

Goals:

Train **text classification** model on English data only
Test on 5+ typologically diverse languages without training data in those languages
Compare different transfer approaches (translate-train, translate-test, multilingual models, adapter-based)
Analyze which linguistic features (word order, morphology, script) affect transfer success
Investigate intermediate language selection strategies

Technical Challenges:

Handle script differences (Latin, Cyrillic, Arabic, Chinese, etc.)
Address vocabulary mismatch across languages
Deal with different linguistic phenomena (case systems, gender, etc.)
Minimize English bias in predictions

Datasets:

XNLI (Natural Language Inference in 15 languages)
MLDoc (document classification in 8 languages)
WikiANN NER (cross-lingual named entity recognition)
PAWS-X (paraphrase identification)

Suggested Approaches:

Fine-tune mBERT or XLM-R on English
Implement language adapters
Use intermediate task transfer
Apply data augmentation with back-translation
Evaluation Metrics: Accuracy per language, performance gap from English, correlation with typological distance

=========================================================================


Training pipeline:
1. Load tokenizer (xlm-roberta-base)
        ↓
2. Load XNLI from data/raw/xnli
        ↓
3. Tokenize train + validation (English only)
        ↓
4. Load model
        ↓
5. Partial fine-tune on English
        ↓
6. Zero-shot evaluate on: en, de, zh, ar, ru, hi
        ↓
7. Print results table with gap from English
        ↓
8. Save adapter to checkpoints/

### To be ran on max locally: 
Copy checkpoint from Google Drive to local:
```bash
cp -r ~/Google\ Drive/My\ Drive/nlp_project/checkpoints/xnli_lora_adapter checkpoints/
```
```bash
Run inference (works fine on Mac MPS!):
python src/inference.py --mode demo
```