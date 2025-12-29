"""
Zero-Shot Cross-Lingual NLI Demo
Interactive web interface for testing multilingual natural language inference.
Train on English → Test on ANY language!
"""

import gradio as gr
from gradio import themes
import torch
import time
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ==================== Configuration ====================
MODEL_NAME = "xlm-roberta-base"
CHECKPOINT_DIR = "checkpoints/xnli_lora_adapter"
NUM_LABELS = 3

LABEL_NAMES = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
LABEL_COLORS = {0: "#4CAF50", 1: "#FFC107", 2: "#F44336"}  # Green, Yellow, Red
LABEL_EMOJIS = {0: "✅", 1: "🤔", 2: "❌"}


LANGUAGES = {
    "English": "en",
    "German": "de", 
    "Chinese": "zh",
    "Arabic": "ar",
    "Russian": "ru",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "Japanese": "ja",
}


EXAMPLES = [
    # English
    ["A man is playing guitar on stage.", "A musician is performing.", "English"],
    ["The cat is sleeping on the couch.", "The animal is resting.", "English"],
    ["It is raining heavily outside.", "The weather is sunny and clear.", "English"],
    
    # German
    ["Ein Mann spielt Gitarre auf der Bühne.", "Ein Musiker tritt auf.", "German"],
    ["Die Katze schläft auf dem Sofa.", "Das Tier ruht sich aus.", "German"],
    
    # Chinese
    ["一个男人在舞台上弹吉他。", "一位音乐家正在表演。", "Chinese"],
    ["猫在沙发上睡觉。", "动物正在休息。", "Chinese"],
    
    # French
    ["Un homme joue de la guitare sur scène.", "Un musicien se produit.", "French"],
    ["Le chat dort sur le canapé.", "L'animal se repose.", "French"],
    
    # Spanish
    ["Un hombre está tocando la guitarra en el escenario.", "Un músico está actuando.", "Spanish"],
    
    # Arabic
    ["رجل يعزف على الجيتار على المسرح.", "موسيقي يؤدي.", "Arabic"],
    
    # Russian
    ["Мужчина играет на гитаре на сцене.", "Музыкант выступает.", "Russian"],
    
    # Japanese
    ["男性がステージでギターを弾いている。", "ミュージシャンが演奏している。", "Japanese"],
]


# ==================== Model Loading ====================
class NLIModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self.model_size_mb = 0
        self.trainable_params = 0
        self.total_params = 0
        
    def load(self):
        """Load the model and tokenizer."""
        print("Loading model...")
        
        # Device detection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        if os.path.exists(CHECKPOINT_DIR):
            print(f"Loading from checkpoint: {CHECKPOINT_DIR}")
            self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
            
            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=NUM_LABELS,
                torch_dtype=torch.float32,
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
        else:
            print(f"Checkpoint not found at {CHECKPOINT_DIR}")
            print("Loading base model without adapter (for demo purposes)...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=NUM_LABELS,
                torch_dtype=torch.float32,
            )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Calculate model statistics
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024 * 1024)
        
        self.model_loaded = True
        print("Model loaded successfully!")
        
    def predict(self, premise, hypothesis):
        """Run inference and return predictions with timing."""
        if not self.model_loaded or self.tokenizer is None or self.model is None:
            return None, None, 0
        
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        inputs = inputs.to(self.device)
        
        # Inference with timing
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        probs = probs[0].cpu().numpy()
        predicted_label = int(probs.argmax())
        
        return predicted_label, probs, inference_time


nli_model = NLIModel()


# ==================== Gradio Interface ====================
def predict_nli(premise: str, hypothesis: str, language: str) -> tuple:
    """
    Main prediction function for Gradio interface.
    Returns: label, confidence chart, metrics HTML
    """
    # Input validation
    if not premise or not premise.strip():
        return (
            "Please enter a premise.",
            None,
            "<p style='color: orange;'>Error: Premise is required.</p>"
        )
    
    if not hypothesis or not hypothesis.strip():
        return (
            "Please enter a hypothesis.",
            None,
            "<p style='color: orange;'>Error: Hypothesis is required.</p>"
        )
    
    # Load model if not loaded
    if not nli_model.model_loaded:
        try:
            nli_model.load()
        except Exception as e:
            return (
                "Model loading failed.",
                None,
                f"<p style='color: red;'>Error loading model: {str(e)}</p>"
            )
    
    # Run prediction
    try:
        predicted_label, probs, inference_time = nli_model.predict(premise.strip(), hypothesis.strip())
        
        if predicted_label is None or probs is None:
            return (
                "Prediction failed.",
                None,
                "<p style='color: red;'>Error during prediction.</p>"
            )
        
        # Format result
        label_name = LABEL_NAMES[predicted_label]
        emoji = LABEL_EMOJIS[predicted_label]
        confidence = probs[predicted_label] * 100
        
        result_text = f"{emoji} **{label_name}** ({confidence:.1f}% confidence)"
        
        # Create confidence chart data
        chart_data = {
            "Label": list(LABEL_NAMES.values()),
            "Confidence": [float(p) * 100 for p in probs],
        }
        
        # Metrics HTML
        metrics_html = f"""
        <div style="padding: 10px; background: #f5f5f5; border-radius: 8px; margin-top: 10px;">
            <h4 style="margin: 0 0 10px 0;"> Performance Metrics</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 5px;"><b>Inference Time:</b></td>
                    <td style="padding: 5px;">{inference_time:.2f} ms</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Device:</b></td>
                    <td style="padding: 5px;">{str(nli_model.device).upper()}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Model Size:</b></td>
                    <td style="padding: 5px;">{nli_model.model_size_mb:.1f} MB</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Total Parameters:</b></td>
                    <td style="padding: 5px;">{nli_model.total_params:,}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>LoRA Parameters:</b></td>
                    <td style="padding: 5px;">{nli_model.trainable_params:,} ({nli_model.trainable_params/nli_model.total_params*100:.2f}%)</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><b>Input Language:</b></td>
                    <td style="padding: 5px;">{language} (zero-shot)</td>
                </tr>
            </table>
        </div>
        """
        
        return result_text, chart_data, metrics_html
        
    except Exception as e:
        return (
            "Error during prediction.",
            None,
            f"<p style='color: red;'>Error: {str(e)}</p>"
        )


def create_demo():
    """Create and return the Gradio demo interface."""
    
    with gr.Blocks(
        title="Zero-Shot Cross-Lingual NLI",
        theme=themes.Soft(),
        css="""
            .main-title { text-align: center; margin-bottom: 20px; }
            .subtitle { text-align: center; color: #666; margin-bottom: 30px; }
            .result-box { font-size: 1.3em; padding: 15px; }
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # Zero-Shot Cross-Lingual Natural Language Inference
            
            <p style="text-align: center; font-size: 1.1em; color: #555;">
                Train on <b>English</b> → Test on <b>ANY language</b>!<br>
                Model: XLM-RoBERTa with LoRA adapters
            </p>
            """,
            elem_classes=["main-title"]
        )
        
        # Main content
        with gr.Row():
            # Left column - Inputs
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                premise_input = gr.Textbox(
                    label="Premise",
                    placeholder="Enter the premise (a factual statement)...",
                    lines=3,
                    elem_id="premise"
                )
                
                hypothesis_input = gr.Textbox(
                    label="Hypothesis", 
                    placeholder="Enter the hypothesis (a claim to evaluate)...",
                    lines=3,
                    elem_id="hypothesis"
                )
                
                language_dropdown = gr.Dropdown(
                    choices=list(LANGUAGES.keys()),
                    value="English",
                    label="Input Language",
                    info="Select the language of your input (for display purposes)"
                )
                
                predict_btn = gr.Button("Analyze", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    ---
                    ### ℹ️ About NLI Labels
                    - ✅ **Entailment**: Hypothesis follows from premise
                    - 🤔 **Neutral**: Hypothesis is unrelated to premise  
                    - ❌ **Contradiction**: Hypothesis conflicts with premise
                    """
                )
            
            # Right column - Results
            with gr.Column(scale=1):
                gr.Markdown("### Results")
                
                result_output = gr.Markdown(
                    value="*Enter text and click Analyze*",
                    elem_classes=["result-box"]
                )
                
                confidence_chart = gr.BarPlot(
                    x="Label",
                    y="Confidence",
                    title="Confidence Scores (%)",
                    height=250,
                    color="Label",
                )
                
                metrics_output = gr.HTML(
                    value="<p style='color: #888;'>Metrics will appear after prediction.</p>"
                )
        
        # Examples section
        gr.Markdown("---")
        gr.Markdown("### Try Examples in Different Languages")
        gr.Markdown("*Click any example below to test zero-shot cross-lingual transfer:*")
        
        gr.Examples(
            examples=EXAMPLES,
            inputs=[premise_input, hypothesis_input, language_dropdown],
            outputs=[result_output, confidence_chart, metrics_output],
            fn=predict_nli,
            cache_examples=False,
        )
        
        # How it works section
        gr.Markdown(
            """
            ---
            ### How It Works
            
            | Component | Description |
            |-----------|-------------|
            | **Base Model** | XLM-RoBERTa (multilingual, 100+ languages) |
            | **Adapter** | LoRA (Low-Rank Adaptation) - only 0.5% parameters trained |
            | **Training Data** | XNLI English only (~392k examples) |
            | **Zero-Shot Transfer** | Model generalizes to unseen languages! |
            
            The multilingual encoder creates a shared representation space where semantically 
            similar sentences from different languages are mapped close together. Training on 
            English teaches the model the NLI task, and this knowledge transfers to other languages.
            """
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            <p style="text-align: center; color: #888; font-size: 0.9em;">
                Built with Transformers, PEFT, and Gradio | 
                Project: Zero-Shot Cross-Lingual Transfer for NLP
            </p>
            """
        )
        
        # Event handlers
        predict_btn.click(
            fn=predict_nli,
            inputs=[premise_input, hypothesis_input, language_dropdown],
            outputs=[result_output, confidence_chart, metrics_output],
        )
        
        # Also trigger on Enter key in hypothesis field
        hypothesis_input.submit(
            fn=predict_nli,
            inputs=[premise_input, hypothesis_input, language_dropdown],
            outputs=[result_output, confidence_chart, metrics_output],
        )
    
    return demo


# ==================== Main ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Zero-Shot Cross-Lingual NLI Demo")
    print("=" * 50)
    
    # Pre-load model
    print("\nPre-loading model...")
    nli_model.load()
    
    # Launch demo
    print("\nLaunching Gradio interface...")
    demo = create_demo()
    demo.launch(
        share=False,  # Set to True to get a public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
