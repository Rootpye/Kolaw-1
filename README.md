# Kolaw-1

---

## Project Overview

Kolaw-1 is a specialized language model developed by fine-tuning **Naver's HyperCLOVAX-SEED-Text-Instruct-0.5B** on an extensive collection of **Korean legal documents**. The primary goal of this project is to enhance the model's understanding and generation capabilities specifically within the legal domain, enabling it to perform tasks such as legal document comprehension, question answering, and summarization with greater accuracy and relevance.

The model leverages the **QLoRA (Quantized Low-Rank Adaptation) technique**, allowing for efficient training even with limited computational resources, specifically within a Google Colab environment.

---

## The Kolaw-1 Model: Specialized for Korean Legal Text

The Kolaw-1 model is not a completely new architecture, but rather an adaptation of the powerful HyperCLOVAX-SEED-Text-Instruct-0.5B base model. Through fine-tuning on a curated dataset of Korean legal texts, it has acquired domain-specific knowledge and stylistic nuances.

**Key characteristics of the Kolaw-1 model:**

* **Base Architecture:** HyperCLOVAX-SEED-Text-Instruct-0.5B (a causal language model, designed for text generation based on preceding text).
* **Domain Specialization:** Trained on various Korean legal documents (excluding specific types like judgments, administrative decisions, and certain court rulings to focus on general legal text, statutes, interpretations, Q&A, and summaries). This specialization allows for more coherent and contextually appropriate responses in legal contexts compared to a general-purpose LLM.
* **Parameter-Efficient Fine-Tuning (PEFT):** Uses LoRA adapters, which means only a small fraction of the model's parameters were trained. This results in a smaller "adapter" file that can be easily merged with the original base model for inference, or further fine-tuned.
* **Quantization (QLoRA):** The base model was loaded in 4-bit quantization during training, significantly reducing its memory footprint. For inference, it's typically loaded with the same quantization or merged into a full-precision model if resources allow.

This model is ideal for applications requiring an understanding of Korean legal terminology, concepts, and typical legal phrasing.

---

## How to Use the Kolaw-1 Model (Inference)

To use the fine-tuned Kolaw-1 model, you'll need to load the original base model and then integrate the trained LoRA adapters. This process merges the specialized knowledge from the fine-tuning into the base model, preparing it for various NLP tasks.

### 1. Environment Setup

You can run the inference on Google Colab, a local GPU machine, or any environment with sufficient GPU memory.

* **Python:** 3.9+ recommended
* **Libraries:** Install the necessary libraries using pip.

    ```bash
    # Ensure you have torch with CUDA support
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124) # Adjust cu124 to your CUDA version if different

    # Install Hugging Face libraries for model loading and PEFT
    pip install transformers peft accelerate bitsandbytes
    ```

### 2. Prepare Your Model Files

Ensure you have the fine-tuned model files from your training process. These would typically be saved in your Google Drive under `Kolaw-1/final_model/` (or the last `checkpoint-XXXX` folder).

**Required files from your trained model directory:**
* `adapter_config.json`
* `adapter_model.safetensors`
* `tokenizer.json` (or other tokenizer files like `tokenizer_config.json`, `special_tokens_map.json`, `vocab.json`)

Make sure these files are accessible in your environment (e.g., copied to your local machine, or mounted from Google Drive in Colab).

### 3. Load the Model and Tokenizer for Inference

The most robust way to use the QLoRA-trained model for inference is to load the original base model and then load your PEFT (LoRA) adapters on top of it, finally merging them.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# Define the original base model checkpoint
base_model_checkpoint = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

# Define the path to your fine-tuned model's directory (where adapter files are saved)
# IMPORTANT: Replace './path/to/your/Kolaw-1_final_model' with the actual path
# If using Colab, this might be '/content/drive/MyDrive/Kolaw-1/final_model'
# or the path to your latest checkpoint folder like '/content/drive/MyDrive/Kolaw-1/checkpoint-1500'
fine_tuned_model_path = "./path/to/your/Kolaw-1_final_model"

# Determine compute dtype (important for consistent behavior with training)
# Use bfloat16 if GPU supports it (NVIDIA Ampere architecture or newer)
compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
print(f"Using compute_dtype: {compute_dtype}")

# 1. Load the tokenizer (from the fine-tuned model path or base model)
# It's generally safer to load the tokenizer from your fine-tuned model's save path
# as it might have added new tokens during training.
tokenizer = AutoTokenizer.from_pretrained(
    fine_tuned_model_path,
    model_max_length=4096,
    padding_side="right", # Recommended for causal language models
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded successfully.")

# 2. Load the base model (in the same compute dtype as training, or float16/float32)
# Using low_cpu_mem_usage=True helps with memory during loading.
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_checkpoint,
    torch_dtype=compute_dtype, # Use the determined compute_dtype
    device_map="auto",
    low_cpu_mem_usage=True # Important for large models
)
print("Base model loaded.")

# 3. Load the PEFT (LoRA) adapters onto the base model
fine_tuned_model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
print("PEFT adapters loaded onto base model.")

# 4. Merge the LoRA adapters into the base model for efficient inference
# This creates a single, merged model suitable for deployment or direct use.
merged_model = fine_tuned_model.merge_and_unload()
print("LoRA adapters merged into the model.")

# Optional: If you want to save this merged model for direct loading later
# merged_model.save_pretrained("./Kolaw-1_merged_model")
# tokenizer.save_pretrained("./Kolaw-1_merged_model")
# print("Merged model saved for direct inference.")

# 5. Create a text generation pipeline
# Ensure `device` is set to 0 for GPU, or -1 for CPU
generator = pipeline(
    "text-generation",
    model=merged_model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1 # Use GPU if available, else CPU
)
print("Text generation pipeline created.")

# Example usage
print("\n--- Generating Text Examples ---")

# Prompt 1: Basic legal statement
prompt_1 = "대한민국 헌법 제1조는" # "The first article of the Constitution of the Republic of Korea states"
generated_text_1 = generator(
    prompt_1,
    max_length=100,
    num_return_sequences=1,
    do_sample=True,      # Enable sampling for more diverse outputs
    top_k=50,            # Consider top 50 most likely tokens
    top_p=0.95,          # Consider tokens whose cumulative probability is 95%
    temperature=0.7,     # Controls randomness of output (lower = less random)
    no_repeat_ngram_size=2 # Prevent repeating 2-gram sequences
)[0]['generated_text']
print(f"\n--- Generated Text 1 ---")
print(generated_text_1)

# Prompt 2: Scenario-based query
prompt_2 = "법원의 판결에 따르면, " # "According to the court's ruling, "
generated_text_2 = generator(
    prompt_2,
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    no_repeat_ngram_size=2
)[0]['generated_text']
print(f"\n--- Generated Text 2 ---")
print(generated_text_2)

# Prompt 3: Specific legal term
prompt_3 = "민법 제750조는" # "Article 750 of the Civil Act states"
generated_text_3 = generator(
    prompt_3,
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    no_repeat_ngram_size=2
)[0]['generated_text']
print(f"\n--- Generated Text 3 ---")
print(generated_text_3)
```

---

### Training Environment
- Hardware Type: NVIDIA T4 16gb
- Hours Used: 16
- Platform: Google Colab (free tier)
