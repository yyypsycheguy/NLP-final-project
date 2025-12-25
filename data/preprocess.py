# This script handles downloading and preprocessing datasets
'''Zero-Shot Cross-Lingual Transfer - Text Classification Datasets
XNLI (Natural Language Inference in 15 languages)
WikiANN NER (cross-lingual named entity recognition)
PAWS-X (paraphrase identification)'''

import os
from datasets import load_dataset, load_from_disk

DATASETS = {
    "xnli": {
        "path": "xnli", 
        "config": "all_languages",
        "task": "Natural Language Inference (3-class classification)"
    },
    "wikiann": {
        "path": "wikiann", 
        "config": "en",  # Load per-language: "en", "de", "zh", "ar", "ru", etc.
        "task": "Named Entity Recognition"
    },
    "paws-x": {
        "path": "google-research-datasets/paws-x", 
        "config": "en",  # Load per-language
        "task": "Paraphrase Identification (binary classification)"
    },
}
TARGET_LANGUAGES = ["en", "de", "zh", "ar", "ru", "hi", "ja"]

# ================ Download datasets =========================
def download_dataset(name, save_dir="raw"):
    """Download a single dataset."""
    config = DATASETS[name]
    save_path = os.path.join(save_dir, name)
    
    if os.path.exists(save_path):
        print(f"[{name}] Already exists, skipping.")
        return None
    
    print(f"[{name}] Downloading: {config['task']}...")
    dset = load_dataset(config["path"], config["config"])
    
    os.makedirs(save_path, exist_ok=True)
    dset.save_to_disk(save_path)
    print(f"[{name}] Saved to {save_path}")
    return dset

# ================ Download Multilanaguage datasets =========================
def download_multilang_dataset(name, languages, save_dir="raw"):
    """Download dataset for multiple languages."""
    config = DATASETS[name]
    
    for lang in languages:
        save_path = os.path.join(save_dir, name, lang)
        
        if os.path.exists(save_path):
            print(f"[{name}/{lang}] Already exists, skipping.")
            continue
        
        print(f"[{name}/{lang}] Downloading...")
        try:
            dset = load_dataset(config["path"], lang)
            os.makedirs(save_path, exist_ok=True)
            dset.save_to_disk(save_path)
            print(f"[{name}/{lang}] Saved to {save_path}")
        except Exception as e:
            print(f"[{name}/{lang}] Error: {e}")


# ================ Tokenizating xlin dataset =========================
def tokenize_xnli(dataset, tokenizer, lang="en", max_length=128):
    """
    Tokenize XNLI dataset for a specific language.
    
    Args:
        dataset: XNLI
        tokenizer: HuggingFace tokenizer
        lang: Language code (en, de, zh, ar, ru, hi, etc.)
        max_length: Maximum sequence length
    
    Returns:
        Tokenized dataset ready for DataLoader
    """
    def preprocess(examples):
        # XNLI stores text as dicts: {"en": "...", "de": "...", ...}
        premises = [p[lang] if isinstance(p, dict) else p for p in examples["premise"]]
        hypotheses = [h[lang] if isinstance(h, dict) else h for h in examples["hypothesis"]]
        
        tokenized = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokenized["labels"] = examples["label"]
        return tokenized
    
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


# ================ Tokenizating pawsx dataset =========================
def tokenize_paws_x(dataset, tokenizer, max_length=128):
    """
    Tokenize PAWS-X dataset (binary paraphrase classification).
    """
    def preprocess(examples):
        tokenized = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokenized["labels"] = examples["label"]
        return tokenized
    
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


# ================ DataLoader =========================
def get_dataloader(tokenized_dataset, batch_size=16, shuffle=True):
    """
    Create a PyTorch DataLoader from tokenized dataset.
    
    Args:
        tokenized_dataset: Dataset with 'input_ids', 'attention_mask', 'labels'
        batch_size: Batch size
        shuffle: Whether to shuffle (True for train, False for eval)
    
    Returns:
        DataLoader ready for training/evaluation
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def prepare_xnli_dataloaders(tokenizer, batch_size=16, train_lang="en", eval_langs=None, data_dir="data/raw/xnli"):
    """
    Prepare train and eval dataloaders for XNLI.
    
    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        train_lang: Language for training (default: "en")
        eval_langs: List of languages for zero-shot evaluation
        data_dir: Path to saved XNLI dataset (uses local if exists, else downloads)
    
    Returns:
        train_loader: DataLoader for training
        eval_loaders: Dict of {lang: DataLoader} for evaluation
    """
    if eval_langs is None:
        eval_langs = ["en", "de", "zh", "ar", "ru", "hi"]
    
    # Load XNLI dataset from local disk if available, else download
    if os.path.exists(data_dir):
        print(f"Loading XNLI from local: {data_dir}")
        dataset = load_from_disk(data_dir)
    else:
        print("Downloading XNLI from HuggingFace...")
        dataset = load_dataset("xnli", "all_languages")
    
    # Tokenize training data (English only)
    train_tokenized = tokenize_xnli(dataset["train"], tokenizer, lang=train_lang)
    train_loader = get_dataloader(train_tokenized, batch_size=batch_size, shuffle=True)
    
    # Tokenize validation data for each language (zero-shot evaluation)
    eval_loaders = {}
    for lang in eval_langs:
        eval_tokenized = tokenize_xnli(dataset["validation"], tokenizer, lang=lang)
        eval_loaders[lang] = get_dataloader(eval_tokenized, batch_size=batch_size, shuffle=False)
    
    return train_loader, eval_loaders


if __name__ == "__main__":
    print("Downloading datasets for Zero-Shot Cross-Lingual Classification...")
    print("-" * 50)
    
    # XNLI has all languages in one config
    download_dataset("xnli")
    # WikiANN and PAWS-X needs to donwload one language at a time
    download_multilang_dataset("paws-x", TARGET_LANGUAGES)


