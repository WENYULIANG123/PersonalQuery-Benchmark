import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Dictionary to cache loaded Hugging Face models and tokenizers
loaded_hf_models = {}


def complete_text_hf(message: str, 
                     model: str = "huggingface/codellama/CodeLlama-7b-hf", 
                     max_tokens: int = 2000, 
                     temperature: float = 0.5, 
                     json_object: bool = False,
                     max_retry: int = 1,
                     sleep_time: int = 0,
                     stop_sequences: list = [], 
                     **kwargs) -> str:
    """
    Generate text completion using a specified Hugging Face model.

    Args:
        message (str): The input text message for completion.
        model (str): The Hugging Face model to use. Default is "huggingface/codellama/CodeLlama-7b-hf".
        max_tokens (int): The maximum number of tokens to generate. Default is 2000.
        temperature (float): Sampling temperature for generation. Default is 0.5.
        json_object (bool): Whether to format the message for JSON output. Default is False.
        max_retry (int): Maximum number of retries in case of an error. Default is 1.
        sleep_time (int): Sleep time between retries in seconds. Default is 0.
        stop_sequences (list): List of stop sequences to halt the generation.
        **kwargs: Additional keyword arguments for the `generate` function.

    Returns:
        str: The generated text completion.
    """
    if json_object:
        message = "You are a helpful assistant designed to output in JSON format." + message
    
    # Determine the device to run the model on (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model.split("/", 1)[1]
    
    # Load the model and tokenizer if not already loaded
    if model_name in loaded_hf_models:
        hf_model, tokenizer = loaded_hf_models[model_name]
    else:
        print(f"🔄 Loading {model_name} to {device}...")
        # Memory-efficient model loading with additional optimizations
        if device.type == "cuda":
            # Pre-clean memory before loading large model
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,  # Use half precision on GPU
            low_cpu_mem_usage=True,
            device_map="auto" if device.type == "cuda" else None,  # Automatic device placement
        )

        # Move to device if not already done by device_map
        if device.type == "cuda" and not hasattr(hf_model, 'hf_device_map'):
            hf_model = hf_model.to(device)

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"⚠️ AutoTokenizer failed: {e}. Trying LlamaTokenizer as fallback...")
            try:
                from transformers import LlamaTokenizer
                tokenizer = LlamaTokenizer.from_pretrained(model_name)
            except (ImportError, Exception):
                # If LlamaTokenizer also fails or is not found in old version
                print("⚠️ LlamaTokenizer also failed. Trying AutoTokenizer with use_fast=False...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        loaded_hf_models[model_name] = (hf_model, tokenizer)

        if device.type == "cuda":
            print(f"✅ Model {model_name} loaded, GPU memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB allocated")
        else:
            print(f"✅ Model {model_name} loaded successfully")
    
    # Encode the input message
    encoded_input = tokenizer(message, return_tensors="pt", return_token_type_ids=False).to(device)
    
    for cnt in range(max_retry):
        try:
            # Generate text completion with memory optimization
            output = hf_model.generate(
                **encoded_input,
                temperature=temperature,
                max_new_tokens=max_tokens,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=False,  # Disable scores to save memory
                pad_token_id=tokenizer.eos_token_id,  # Proper padding
                **kwargs,
            )
            # Decode the generated sequences
            sequences = output.sequences
            sequences = [sequence[len(encoded_input.input_ids[0]):] for sequence in sequences]
            all_decoded_text = tokenizer.batch_decode(sequences)
            completion = all_decoded_text[0]
            return completion
        except Exception as e:
            print(f"Retry {cnt}: {e}")
            time.sleep(sleep_time)
    
    raise RuntimeError("Failed to generate text completion after max retries")
