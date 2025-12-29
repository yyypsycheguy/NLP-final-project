import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


# ================ Load model =========================
def load_xlm_roberta_base_model(model_name="xlm-roberta-base", num_labels=3):

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        dtype=torch.float32,  # float32 for Mac compatibility
    )

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Unfreeze the last 3 encoder layers (layers 9, 10, 11 for XLM-RoBERTa-base)
    for layer in model.roberta.encoder.layer[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    model = model.to(device)
    
    # Print trainable parameters info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")

    return model, tokenizer



# ================ Finetuning model =========================
def finetune_model(model, train_dataset, val_dataset=None, config=None):
    """
    Fine-tune the partially unfrozen model on training data.
    
    Args:
        model: Partially frozen model (classifier + last 3 layers unfrozen)
        train_dataset: Tokenized dataset with 'input_ids', 'attention_mask', 'labels'
        val_dataset: Optional validation dataset
        config: Dict to override default hyperparameters
    
    Returns:
        model: Fine-tuned model
    """
    default_config = {
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,  # Lower LR for partial fine-tuning
        'warmup_ratio': 0.1,
        'weight_decay': 0.01,
    }
    if config is not None:
        default_config.update(config)
    cfg = default_config
    
    # Calculate training steps
    num_batches = (len(train_dataset) + cfg['batch_size'] - 1) // cfg['batch_size']
    total_num_training_steps = num_batches * cfg['epochs']
    num_warmup_steps = int(total_num_training_steps * cfg['warmup_ratio'])

    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=cfg['batch_size']
    )

    model = model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],
    )

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_num_training_steps
    )

    print(f"\n{'='*50}")
    print(f"Starting fine-tuning...")
    print(f"  Device: {device}")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Learning rate: {cfg['learning_rate']}")
    print(f"  Total steps: {total_num_training_steps}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*50}\n")

    best_val_accuracy = 0
    loss_history = []
    val_acc_history = []
    
    for epoch in range(cfg['epochs']):
        # training 
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_loss / len(train_dataloader)
        loss_history.append(avg_train_loss)
        
        # training validation 
        if val_dataset is not None:
            val_accuracy = evaluate(model, val_dataset, cfg['batch_size'])
            val_acc_history.append(val_accuracy)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Accuracy = {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                print(f"  ✓ New best accuracy!")
        else:
            val_acc_history.append(0.0)  # placeholder
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")
    
    print(f"\nTraining complete!")
    if val_dataset is not None:
        print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    return model, loss_history, val_acc_history



# ================ Validation on different language =========================
def evaluate(model, dataset, batch_size=16):
    """
    Evaluate model accuracy on a dataset.
    
    Use this for:
    - Validation during training (English)
    - Zero-shot evaluation on other languages
    
    Args:
        model: The model to evaluate
        dataset: Tokenized dataset with 'input_ids', 'attention_mask', 'labels'
        batch_size: Batch size for evaluation
    
    Returns:
        accuracy: Float between 0 and 1
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
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



# ================ Model inference =========================
def zero_shot_prediction(model, tokenizer, text, device="cpu"):
    """Predict class for input text."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1)
    
    return prediction.item()
