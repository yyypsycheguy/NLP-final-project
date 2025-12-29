import os
import sys
import json
import torch
import pandas as pd
from collections import defaultdict
import argparse

# === Colab/Local Environment Setup ===
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from IPython.display import display, HTML
    print("Running in Google Colab. If using data from Google Drive, run:")
    print("!pip install -q peft transformers datasets")
    print("from google.colab import drive; drive.mount('/content/drive')")
    print("Set data_dir to your drive path, e.g. '/content/drive/MyDrive/your_project/data/raw/xnli'")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader

# ==================== Configuration ====================
MODEL_NAME = "xlm-roberta-base"
CHECKPOINT_DIR = "checkpoints/xnli_lora_adapter"
NUM_LABELS = 3

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
EVAL_LANGS = ["en", "de", "zh", "ar", "ru", "hi"]  # Without French

# Language metadata for analysis
LANGUAGE_INFO = {
    "en": {"name": "English", "family": "Germanic", "script": "Latin", "morphology": "Low", "word_order": "SVO"},
    "de": {"name": "German", "family": "Germanic", "script": "Latin", "morphology": "Medium", "word_order": "V2/SOV"},
    "zh": {"name": "Chinese", "family": "Sino-Tibetan", "script": "Hanzi", "morphology": "Low", "word_order": "SVO"},
    "ar": {"name": "Arabic", "family": "Semitic", "script": "Arabic", "morphology": "High", "word_order": "VSO"},
    "ru": {"name": "Russian", "family": "Slavic", "script": "Cyrillic", "morphology": "High", "word_order": "Free"},
    "hi": {"name": "Hindi", "family": "Indo-Aryan", "script": "Devanagari", "morphology": "High", "word_order": "SOV"},
}

# Error categories (5 well-defined types)
ERROR_CATEGORIES = {
    "E1": "Lexical Overlap – High word similarity led to wrong Entailment prediction",
    "E2": "Negation Failure – Missed negation markers (not, لا, 不, nicht, etc.)",
    "E3": "Script/Tokenization – Subword splits disrupted semantic meaning",
    "E4": "Morphological – Case endings or inflections caused misinterpretation",
    "E5": "Word Order – Non-SVO syntax confused the English-trained model",
}


# Device detection (Colab: CUDA, Mac: MPS, fallback: CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
    device_name = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device('mps')
    device_name = 'mps'
else:
    device = torch.device('cpu')
    device_name = 'cpu'
print(f"Using device: {device_name}")


# ==================== Model Loading ====================

def load_model(baseline=False):
    """Load the trained model with LoRA adapter, or baseline if specified."""
    if baseline:
        print(f"Loading baseline model from HuggingFace: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            torch_dtype=torch.float32,
        )
        model = model.to(device)
        model.eval()
        return model, tokenizer
    else:
        print(f"Loading fine-tuned model from: {CHECKPOINT_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
        model = model.to(device)
        model.eval()
        return model, tokenizer


# ==================== Error Collection ====================
def collect_errors(model, tokenizer, dataset, lang, max_errors=15):
    """
    Collect misclassified examples for a specific language.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        dataset: XNLI validation split
        lang: Language code
        max_errors: Maximum errors to collect per language
    
    Returns:
        List of error dictionaries
    """
    errors = []
    total_checked = 0
    
    print(f"  Collecting errors for {lang}...")
    
    for example in dataset:
        # Extract text for this language
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        
        # Handle XNLI's multilingual dict format
        if isinstance(premise, dict):
            premise = premise.get(lang, "")
            hypothesis = hypothesis.get(lang, "")
        
        if not premise or not hypothesis:
            continue
            
        gold_label = example["label"]
        
        # Get model prediction
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(probs, dim=-1).item()
        
        total_checked += 1
        
        # Collect if misclassified
        if pred_label != gold_label:
            errors.append({
                "id": len(errors) + 1,
                "language": lang,
                "language_name": LANGUAGE_INFO[lang]["name"],
                "family": LANGUAGE_INFO[lang]["family"],
                "script": LANGUAGE_INFO[lang]["script"],
                "morphology": LANGUAGE_INFO[lang]["morphology"],
                "word_order": LANGUAGE_INFO[lang]["word_order"],
                "premise": premise,
                "hypothesis": hypothesis,
                "gold_label": LABEL_NAMES[gold_label],
                    "predicted_label": LABEL_NAMES[int(pred_label)],
                    "confidence": float(probs.squeeze()[int(pred_label)]),
                "probs": {
                    "entailment": float(probs[0][0]),
                    "neutral": float(probs[0][1]),
                    "contradiction": float(probs[0][2]),
                },
                "error_category": "",  # To be filled manually
                "notes": "",  # For qualitative analysis
            })
            
            if len(errors) >= max_errors:
                break
    
    print(f"    Found {len(errors)} errors (checked {total_checked} examples)")
    return errors


def collect_all_errors(model, tokenizer, dataset, errors_per_lang=15):
    """Collect errors from all evaluation languages."""
    all_errors = []
    
    for lang in EVAL_LANGS:
        errors = collect_errors(model, tokenizer, dataset, lang, errors_per_lang)
        all_errors.extend(errors)
    
    return all_errors


# ==================== Analysis ====================
def analyze_errors(errors):
    """Generate summary statistics from collected errors."""
    if not errors:
        print("No errors to analyze.")
        return {}
    
    df = pd.DataFrame(errors)
    
    # Summary by language
    print("\n" + "=" * 60)
    print("Error Summary by Language")
    print("=" * 60)
    
    lang_summary = df.groupby("language_name").agg({
        "id": "count",
        "confidence": "mean",
    }).rename(columns={"id": "count", "confidence": "avg_confidence"})
    print(lang_summary.to_string())
    
    # Summary by script type
    print("\n" + "=" * 60)
    print("Error Summary by Script Type")
    print("=" * 60)
    
    script_summary = df.groupby("script").agg({
        "id": "count",
    }).rename(columns={"id": "count"})
    print(script_summary.to_string())
    
    # Confusion matrix (gold vs predicted)
    print("\n" + "=" * 60)
    print("Confusion Pattern (Gold → Predicted)")
    print("=" * 60)
    
    confusion = df.groupby(["gold_label", "predicted_label"]).size().unstack(fill_value=0)
    print(confusion.to_string())
    
    # Summary by morphological complexity
    print("\n" + "=" * 60)
    print("Error Summary by Morphological Complexity")
    print("=" * 60)
    
    morph_summary = df.groupby("morphology").agg({
        "id": "count",
    }).rename(columns={"id": "count"})
    print(morph_summary.to_string())
    
    return {
        "by_language": lang_summary.to_dict(),
        "by_script": script_summary.to_dict(),
        "by_morphology": morph_summary.to_dict(),
        "confusion": confusion.to_dict(),
    }


# ==================== Export ====================
def export_errors(errors, output_dir="error_analysis"):
    """Export errors to CSV and JSON for manual annotation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to CSV (for spreadsheet annotation)
    csv_path = os.path.join(output_dir, "errors_for_annotation.csv")
    df = pd.DataFrame(errors)
    
    # Reorder columns for easier annotation
    columns = [
        "id", "language", "language_name", "premise", "hypothesis",
        "gold_label", "predicted_label", "confidence",
        "error_category", "notes",
        "family", "script", "morphology"
    ]
    df = df[[c for c in columns if c in df.columns]]
    df.to_csv(csv_path, index=False)
    print(f"\nExported {len(errors)} errors to: {csv_path}")
    
    # Export to JSON (for programmatic use)
    json_path = os.path.join(output_dir, "errors_full.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"Exported full data to: {json_path}")
    
    # Print annotation instructions
    print("\n" + "=" * 60)
    print("ANNOTATION INSTRUCTIONS")
    print("=" * 60)
    print("1. Open errors_for_annotation.csv in Excel/Sheets")
    print("2. For each error, fill in 'error_category' with one of:")
    for code, desc in ERROR_CATEGORIES.items():
        print(f"   {code}: {desc}")
    print("3. Add notes explaining the error in 'notes' column")
    print("4. Save and re-run analysis with annotated file")
    
    return csv_path, json_path


def load_annotated_errors(csv_path):
    """Load annotated errors from CSV."""
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def analyze_annotated_errors(csv_path):
    """Analyze errors after manual annotation."""
    df = pd.read_csv(csv_path)
    
    if "error_category" not in df.columns or df["error_category"].isna().all():
        print("No annotations found. Please annotate errors_for_annotation.csv first.")
        return
    
    print("\n" + "=" * 60)
    print("Error Category Distribution")
    print("=" * 60)
    
    category_counts = df["error_category"].value_counts()
    total = len(df)
    
    for cat, count in category_counts.items():
        pct = count / total * 100
        desc = ERROR_CATEGORIES.get(str(cat), "Unknown")
        print(f"{cat}: {count:3d} ({pct:5.1f}%) - {desc}")
    
    print("\n" + "=" * 60)
    print("Error Categories by Language")
    print("=" * 60)
    
    cat_by_lang = pd.crosstab(df["language_name"], df["error_category"])
    print(cat_by_lang.to_string())
    
    print("\n" + "=" * 60)
    print("Error Categories by Script")
    print("=" * 60)
    
    cat_by_script = pd.crosstab(df["script"], df["error_category"])
    print(cat_by_script.to_string())
    
    return cat_by_lang, cat_by_script


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Lingual Error Analysis")
    parser.add_argument('--baseline', action='store_true', help='Run analysis on baseline (not fine-tuned) model')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to export errors/results')
    args = parser.parse_args()

    print("=" * 60)
    print("Zero-Shot Cross-Lingual Error Analysis")
    print("=" * 60)

    # Set output directory based on model type
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = "error_analysis_baseline" if args.baseline else "error_analysis"

    # Check if already annotated
    annotated_csv = os.path.join(output_dir, "errors_for_annotation.csv")
    if os.path.exists(annotated_csv):
        df = pd.read_csv(annotated_csv)
        if "error_category" in df.columns and not df["error_category"].isna().all():
            print("\nFound annotated errors! Analyzing...")
            analyze_annotated_errors(annotated_csv)
            return

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model(baseline=args.baseline)


    # Load dataset (prefer local folder, fallback to HuggingFace)
    print("\nLoading XNLI dataset...")
    data_dir = "data/raw/xnli"
    if IN_COLAB:
        # In Colab, user may need to mount Google Drive and set data_dir accordingly
        if not os.path.exists(data_dir):
            print(f"[Colab] data_dir '{data_dir}' not found. Please mount Google Drive and set the correct path.")
            print("Falling back to HuggingFace datasets...")
            dataset = load_dataset("xnli", "all_languages")
        else:
            dataset = load_from_disk(data_dir)
    else:
        if os.path.exists(data_dir):
            dataset = load_from_disk(data_dir)
        else:
            print(f"[Local] data_dir '{data_dir}' not found. Falling back to HuggingFace datasets...")
            dataset = load_dataset("xnli", "all_languages")

    # Collect errors (15 per language = 105 total)
    print("\nCollecting errors...")
    errors = collect_all_errors(
        model, tokenizer,
        dataset["validation"],
        errors_per_lang=15
    )

    print(f"\nTotal errors collected: {len(errors)}")

    # Analyze
    analyze_errors(errors)

    # Export for annotation
    export_errors(errors, output_dir=output_dir)


if __name__ == "__main__":
    main()
