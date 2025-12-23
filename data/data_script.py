# This script handles downloading and preprocessing datasets
'''Zero-Shot Cross-Lingual Transfer - Text Classification Datasets
XNLI (Natural Language Inference in 15 languages)
WikiANN NER (cross-lingual named entity recognition)
PAWS-X (paraphrase identification)'''

import os
from datasets import load_dataset

# Datasets for text classification zero-shot transfer
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


if __name__ == "__main__":
    print("Downloading datasets for Zero-Shot Cross-Lingual Classification...")
    print("-" * 50)
    
    # XNLI has all languages in one config
    download_dataset("xnli")
    
    # WikiANN and PAWS-X need per-language downloads
    download_multilang_dataset("wikiann", TARGET_LANGUAGES)
    download_multilang_dataset("paws-x", TARGET_LANGUAGES)


