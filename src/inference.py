import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from datasets import load_from_disk
from data.preprocess import tokenize_xnli

# Device detection
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# XNLI label mapping
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_trained_model(checkpoint_dir="checkpoints/xnli_lora_adapter", num_labels=3):
    """
    Load a trained model with LoRA adapter.
    
    Args:
        checkpoint_dir: Path to saved adapter
        num_labels: Number of classification labels
    
    Returns:
        model, tokenizer
    """
    print(f"Loading model from: {checkpoint_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    
    # Load base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=num_labels,
        torch_dtype=torch.float32,
    )
    
    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer


def predict_single(model, tokenizer, premise, hypothesis):
    """
    Predict NLI label for a single premise-hypothesis pair.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        premise: Premise text (any language)
        hypothesis: Hypothesis text (any language)
    
    Returns:
        label: Predicted label (entailment/neutral/contradiction)
        probs: Probability distribution over classes
    """
    model.eval()
    
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
        prediction = torch.argmax(probs, dim=-1).item()
    
    return LABEL_NAMES[prediction], probs[0].cpu().tolist()


def predict_batch(model, tokenizer, examples, batch_size=16):
    """
    Predict NLI labels for a batch of examples.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        examples: List of (premise, hypothesis) tuples
        batch_size: Batch size
    
    Returns:
        predictions: List of predicted labels
    """
    from torch.utils.data import DataLoader
    
    model.eval()
    predictions = []
    
    # Tokenize all examples
    premises = [ex[0] for ex in examples]
    hypotheses = [ex[1] for ex in examples]
    
    inputs = tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    
    # Create simple dataset
    dataset = torch.utils.data.TensorDataset(
        inputs["input_ids"],
        inputs["attention_mask"]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device)
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            predictions.extend([LABEL_NAMES[p.item()] for p in preds])
    
    return predictions


def evaluate_on_language(model, tokenizer, lang="en", data_dir="data/raw/xnli", batch_size=16):
    """
    Evaluate model on a specific language from XNLI.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        lang: Language code (en, de, zh, ar, ru, hi, etc.)
        data_dir: Path to XNLI dataset
        batch_size: Batch size
    
    Returns:
        accuracy: Float between 0 and 1
    """
    from torch.utils.data import DataLoader
    
    print(f"Evaluating on {lang}...")
    
    # Load and tokenize dataset
    dataset = load_from_disk(data_dir)
    eval_tokenized = tokenize_xnli(dataset["validation"], tokenizer, lang=lang)
    
    dataloader = DataLoader(eval_tokenized, batch_size=batch_size)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += len(batch["labels"])
    
    accuracy = correct / total
    return accuracy


def run_zero_shot_evaluation(checkpoint_dir="checkpoints/xnli_lora_adapter", 
                              languages=None, 
                              data_dir="data/raw/xnli"):
    """
    Run zero-shot evaluation across multiple languages.
    """
    if languages is None:
        languages = ["en", "de", "zh", "ar", "ru", "hi"]
    
    model, tokenizer = load_trained_model(checkpoint_dir)
    
    print("\n" + "=" * 50)
    print("Zero-Shot Cross-Lingual Evaluation")
    print("=" * 50)
    
    results = {}
    for lang in languages:
        acc = evaluate_on_language(model, tokenizer, lang, data_dir)
        results[lang] = acc
        print(f"  {lang}: {acc:.4f}")
    
    # Summary
    print("\n" + "-" * 30)
    en_acc = results.get("en", 0)
    print(f"{'Language':<10} {'Accuracy':<10} {'Gap':<10}")
    for lang, acc in results.items():
        gap = en_acc - acc
        print(f"{lang:<10} {acc:.4f}     {gap:+.4f}")
    
    avg = sum(results.values()) / len(results)
    print(f"\nAverage: {avg:.4f}")
    
    return results


def interactive_demo():
    """
    Interactive demo for zero-shot prediction.
    """
    print("\n" + "=" * 50)
    print("Zero-Shot Cross-Lingual NLI Demo")
    print("=" * 50)
    print("Enter premise and hypothesis in ANY language.")
    print("The model was trained on English only!\n")
    
    model, tokenizer = load_trained_model()
    
    while True:
        print("-" * 40)
        premise = input("Premise (or 'quit'): ").strip()
        if premise.lower() == 'quit':
            break
        
        hypothesis = input("Hypothesis: ").strip()
        if not hypothesis:
            continue
        
        label, probs = predict_single(model, tokenizer, premise, hypothesis)
        
        print(f"\nPrediction: {label}")
        print(f"  entailment:    {probs[0]:.3f}")
        print(f"  neutral:       {probs[1]:.3f}")
        print(f"  contradiction: {probs[2]:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Lingual Inference")
    parser.add_argument("--mode", choices=["demo", "eval"], default="demo",
                        help="Mode: 'demo' for interactive, 'eval' for evaluation")
    parser.add_argument("--checkpoint", default="checkpoints/xnli_lora_adapter",
                        help="Path to saved model checkpoint")
    parser.add_argument("--langs", nargs="+", default=["en", "de", "zh", "ar", "ru", "hi"],
                        help="Languages to evaluate")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        interactive_demo()
    else:
        run_zero_shot_evaluation(args.checkpoint, args.langs)