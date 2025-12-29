import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.xlm_roberta_base import (
    load_xlm_roberta_base_model,
    finetune_model,
    evaluate
)
from data.preprocess import (
    tokenize_xnli,
    prepare_xnli_dataloaders
)
from datasets import load_from_disk
from transformers import AutoTokenizer

# Device handling is done internally in models/xlm_roberta_base.py

CONFIG = {
    "model_name": "xlm-roberta-base",
    "num_labels": 3,  # XNLI: entailment, neutral, contradiction
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-4,  # Higher LR for LoRA (only adapter params)
    "epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "train_lang": "en",  # Train ONLY on English
    "eval_langs": ["en", "de", "zh", "ar", "ru", "hi"],
    "data_dir": "data/raw/xnli",
    "checkpoint_dir": "checkpoints",
}


def main():
    print("Zero-Shot Cross-Lingual Transfer - XNLI Training")
    
    # ================ Load Tokenizer =========================
    print(f"\nLoading tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    
    # ================ Load Dataset =========================
    print(f"\nLoading XNLI dataset from: {CONFIG['data_dir']}")
    dataset = load_from_disk(CONFIG["data_dir"])
    print(f"  Train: {len(dataset['train']):,} examples")
    print(f"  Validation: {len(dataset['validation']):,} examples")
    
    # ================ Tokenize Data =========================
    print(f"\nTokenizing training data (lang={CONFIG['train_lang']})...")
    train_tokenized = tokenize_xnli(
        dataset["train"], 
        tokenizer, 
        lang=CONFIG["train_lang"],
        max_length=CONFIG["max_length"]
    )
    
    print(f"Tokenizing validation data (lang={CONFIG['train_lang']})...")
    val_tokenized = tokenize_xnli(
        dataset["validation"], 
        tokenizer, 
        lang=CONFIG["train_lang"],
        max_length=CONFIG["max_length"]
    )
    
    # ================ Load Model =========================
    print(f"\nLoading model: {CONFIG['model_name']}")
    model, _ = load_xlm_roberta_base_model(
        model_name=CONFIG["model_name"], 
        num_labels=CONFIG["num_labels"]
    )
    

    # ================ Fine-tune on English =========================
    print("\nStarting fine-tuning (English only)...")
    model = finetune_model(
        model, 
        train_tokenized, 
        val_tokenized, 
        config={
            "epochs": CONFIG["epochs"],
            "batch_size": CONFIG["batch_size"],
            "learning_rate": CONFIG["learning_rate"],
            "warmup_ratio": CONFIG["warmup_ratio"],
            "weight_decay": CONFIG["weight_decay"],
        }
    )
    
    # ================ Zero-Shot Evaluation =========================
    print("\n" + "=" * 60)
    print("Zero-Shot Cross-Lingual Evaluation")
    print("=" * 60)
    
    results = {}
    for lang in CONFIG["eval_langs"]:
        print(f"\nEvaluating on {lang}...")
        eval_tokenized = tokenize_xnli(
            dataset["validation"], 
            tokenizer, 
            lang=lang,
            max_length=CONFIG["max_length"]
        )
        accuracy = evaluate(model, eval_tokenized, batch_size=CONFIG["batch_size"])
        results[lang] = accuracy
        print(f"  {lang}: {accuracy:.4f}")
    
    # ================ Results Summary =========================
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Language':<10} {'Accuracy':<10} {'Gap from EN':<10}")
    print("-" * 30)
    
    en_acc = results.get("en", 0)
    for lang, acc in results.items():
        gap = en_acc - acc
        print(f"{lang:<10} {acc:.4f}     {gap:+.4f}")
    
    print("-" * 30)
    avg_acc = sum(results.values()) / len(results)
    print(f"{'Average':<10} {avg_acc:.4f}")
    
    # ================ Save Model =========================
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    save_path = os.path.join(CONFIG["checkpoint_dir"], "xnli_lora_adapter")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\nModel saved to: {save_path}")


if __name__ == "__main__":
    main()

