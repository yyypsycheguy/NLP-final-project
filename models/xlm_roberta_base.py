import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_xlm_roberta_base_model(model_name="xlm-roberta-base", num_labels=3):
    """
    Load XLM-RoBERTa with LoRA adapters for sequence classification.
    
    Args:
        model_name: HuggingFace model ID
        num_labels: Number of classification labels (3 for XNLI)
    
    Returns:
        model: PEFT-wrapped model with LoRA adapters
        tokenizer: Corresponding tokenizer
    """
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
        target_modules=["query", "value"]  # adapt attention layers
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def finetune_model(model, train_dataset, task="classification"):
    # Placeholder for fine-tuning logic
    pass


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