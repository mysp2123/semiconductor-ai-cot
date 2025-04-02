"""
Model manager for LLaMA 3 - 70B and Gemini 1.5 Flash.
"""

import os
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure API keys
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Paths for LLaMA 3 - 70B
llama_model_path = os.getenv("LLAMA_MODEL_PATH")
llama_tokenizer_path = os.getenv("LLAMA_TOKENIZER_PATH")

print(f"LLAMA_MODEL_PATH: {llama_model_path}")
print(f"LLAMA_TOKENIZER_PATH: {llama_tokenizer_path}")

# GPU memory configuration
SYSTEM_RESERVE = 2.5  # GB reserved for system
MAX_GPU_MEMORY = 47.5  # GB maximum GPU memory for the model

def clear_memory():
    """Free GPU and CPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Memory cache cleared")

def load_llama_model():
    """Load the LLaMA 3 - 70B model."""
    print("⏳ Loading LLaMA 3 - 70B model...")
    clear_memory()

    tokenizer = AutoTokenizer.from_pretrained(
        llama_tokenizer_path,
        use_fast=True,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        llama_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        max_memory={f"cuda:{i}": f"{int(MAX_GPU_MEMORY)}GiB" for i in range(torch.cuda.device_count())}
    )
    print("✅ LLaMA 3 - 70B model loaded")
    return tokenizer, model

def load_gemini_model():
    """Load the Gemini 1.5 Flash model."""
    print("⏳ Loading Gemini 1.5 Flash model...")
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Gemini 1.5 Flash model loaded")
        return model
    except Exception as e:
        print(f"❌ Error loading Gemini model: {e}")
        return None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def generate_with_llama(prompt, max_tokens=1024, temperature=0.7):
    """Generate text using the LLaMA 3 - 70B model."""
    tokenizer, model = load_llama_model()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
def generate_with_gemini(prompt, max_tokens=1024, temperature=0.7):
    """Generate text using the Gemini 1.5 Flash model."""
    model = load_gemini_model()
    if not model:
        return "[Error: Gemini model could not be loaded]"

    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature
            }
        )
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error generating with Gemini: {e}")
        return f"[Error: {e}]"

def generate_text(prompt, model_type="llama", max_tokens=1024, temperature=0.7):
    """Generate text using the specified model."""
    if model_type.lower() == "llama":
        return generate_with_llama(prompt, max_tokens, temperature)
    elif model_type.lower() == "gemini":
        return generate_with_gemini(prompt, max_tokens, temperature)
    else:
        return "[Error: Unsupported model type. Use 'llama' or 'gemini']"

if __name__ == "__main__":
    # Example usage
    prompt = "Explain the impact of AI on semiconductor research."
    
    print("\n--- Generating with LLaMA 3 - 70B ---")
    llama_response = generate_text(prompt, model_type="llama", max_tokens=200)
    print(llama_response)

    print("\n--- Generating with Gemini 1.5 Flash ---")
    gemini_response = generate_text(prompt, model_type="gemini", max_tokens=200)
    print(gemini_response)