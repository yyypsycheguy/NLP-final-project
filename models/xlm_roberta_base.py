import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def load_xlm_roberta_base_model(model_name="xlm-roberta-base", num_labels=3):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float32,  # float32 for Mac compatibility
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,  # adapter rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"] 
    )
    
    model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

    return model, tokenizer, lora_config


def finetune_model(model, train_dataset, val_dataset=None, task="classification"):
    """
    Fine-tune the LoRA adapter on training data.
    
    Args:
        model: PEFT-wrapped model (already has LoRA adapters from load_xlm_roberta_base_model)
        train_dataset: Tokenized training dataset with 'input_ids', 'attention_mask', 'labels'
        val_dataset: Optional validation dataset
        task: Task type (for logging)
    
    Returns:
        model: Fine-tuned model
    """
    epochs = 3
    batch_size = 16
    learning_rate = 2e-4  # 0.0002 - good for LoRA
    warmup_ratio = 0.1
    
    # Calculate training steps
    num_batches = (len(train_dataset) + batch_size - 1) // batch_size
    total_num_training_steps = num_batches * epochs
    num_warmup_steps = int(total_num_training_steps * warmup_ratio)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Model is already wrapped with PEFT from load_xlm_roberta_base_model
    # No need to call get_peft_model again!
    model = model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_num_training_steps
    )

    print(f"\n{'='*50}")
    print(f"Starting {task} fine-tuning...")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total steps: {total_num_training_steps}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*50}\n")

    best_val_accuracy = 0
    
    for epoch in range(epochs):
        # ============ Training ============
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
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
        
        avg_train_loss = total_loss / len(train_dataloader)
        
        # ============ Validation ============
        if val_dataloader is not None:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    correct += (predictions == batch["labels"]).sum().item()
                    total += len(batch["labels"])
            
            val_accuracy = correct / total
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"  ✓ New best accuracy!")
        else:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
    
    print(f"\nTraining complete!")
    if val_dataloader is not None:
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    return model


def zero_shot_prediction(model, tokenizer, text, device="cpu"):
    """Predict class for input text."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
    
    return prediction.item()


if __name__ == "__main__":
    model, tokenizer = load_xlm_roberta_base_model()
    
    # Test with French text (zero-shot, model trained on English)
    text = "Bonjour, je suis un modèle de langue."
    pred = zero_shot_prediction(model, tokenizer, text)
    print(f"Prediction: {pred}")