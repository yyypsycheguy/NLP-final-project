"""
Training script for Zero-Shot Cross-Lingual Transfer
Fine-tune XLM-RoBERTa + LoRA adapter on English XNLI data only.
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorWithPadding, get_scheduler
from datasets import load_from_disk
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.xlm_roberta_base import load_xlm_roberta_base_model


# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    "model_name": "xlm-roberta-base",
    "num_labels": 3,  # XNLI: entailment, neutral, contradiction
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-4,  # Higher LR for LoRA (only adapter params)
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "train_lang": "en",  # Train ONLY on English
    "data_dir": "data/raw/xnli",
    "checkpoint_dir": "checkpoints",
}


# ============================================================================
# Data Preprocessing
# ============================================================================
def preprocess_function(examples, tokenizer, lang="en", max_length=128):
    """Tokenize premise-hypothesis pairs for a specific language."""
    premises = []
    hypotheses = []
    
    for p, h in zip(examples["premise"], examples["hypothesis"]):
        # Handle multilingual dict format
        if isinstance(p, dict):
            p = p.get(lang, "")
        if isinstance(h, dict):
            h = h.get(lang, "")
        premises.append(str(p))
        hypotheses.append(str(h))
    
    tokenized = tokenizer(
        premises,
        hypotheses,
        truncation=True,
        max_length=max_length,
        padding=False,  # Dynamic padding via DataCollator
    )
    
    return tokenized


def prepare_dataloader(dataset, tokenizer, lang, batch_size, max_length, shuffle=True):
    """Prepare DataLoader with dynamic padding."""
    
    # Tokenize dataset
    tokenized = dataset.map(
        lambda x: preprocess_function(x, tokenizer, lang, max_length),
        batched=True,
        remove_columns=["premise", "hypothesis"],
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    
    # DataCollator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator,
    )


# ============================================================================
# Training Loop
# ============================================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device):
    """Evaluate model on a dataset."""
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


# ============================================================================
# Main Training Function
# ============================================================================
def train(config=None):
    """Main training function."""
    if config is None:
        config = CONFIG
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load model with LoRA adapters
    print("Loading model with LoRA adapters...")
    model, tokenizer = load_xlm_roberta_base_model(
        model_name=config["model_name"],
        num_labels=config["num_labels"],
    )
    model.to(device)
    
    # Load dataset
    print(f"Loading XNLI dataset from {config['data_dir']}...")
    dataset = load_from_disk(config["data_dir"])
    
    # Prepare DataLoaders (English only for training)
    train_loader = prepare_dataloader(
        dataset["train"],
        tokenizer,
        lang=config["train_lang"],
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        shuffle=True,
    )
    
    val_loader = prepare_dataloader(
        dataset["validation"],
        tokenizer,
        lang=config["train_lang"],
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        shuffle=False,
    )
    
    # Optimizer (only LoRA parameters are trainable)
    optimizer = AdamW(
        model.parameters(),  # Only trainable params (LoRA) will be updated
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * config["num_epochs"]
    num_warmup_steps = int(num_training_steps * config["warmup_ratio"])
    
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Training loop
    print("\n" + "=" * 50)
    print("Starting training (English only)...")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 50 + "\n")
    
    best_accuracy = 0
    
    for epoch in range(config["num_epochs"]):
        # Train
        avg_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Evaluate on English validation
        val_accuracy = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Accuracy (EN): {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            os.makedirs(config["checkpoint_dir"], exist_ok=True)
            
            # Save only the LoRA adapter (much smaller than full model!)
            model.save_pretrained(os.path.join(config["checkpoint_dir"], "best_adapter"))
            tokenizer.save_pretrained(os.path.join(config["checkpoint_dir"], "best_adapter"))
            print(f"  ✓ Saved best model (accuracy: {val_accuracy:.4f})")
    
    print("\n" + "=" * 50)
    print(f"Training complete! Best validation accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {config['checkpoint_dir']}/best_adapter")
    print("=" * 50)
    
    return model, tokenizer


if __name__ == "__main__":
    train()
