import torch
from typing import List, Union
from transformers import AutoTokenizer, AutoModel

loaded_gte_models = {}


def get_gte_qwen_embeddings(text: Union[str, List[str]],
                           model_name: str = 'alibaba-nlp/gte-base-en-v1.5',
                           device: str = 'cuda',
                           max_length: int = 512,  # Reduced for speed optimization
                           **kwargs) -> torch.Tensor:
    """
    Get GTE-Qwen2-1.5B-Instruct embeddings for the given text.

    Args:
        text (Union[str, List[str]]): The input text to be embedded.
        model_name (str): The model to use for embedding.
        device (str): Device to run the model on.
        max_length (int): Maximum sequence length.

    Returns:
        torch.Tensor: The embedding(s) of the input text(s).
    """

    if model_name in loaded_gte_models:
        tokenizer, model = loaded_gte_models[model_name]
    else:
        # Pre-load config and patch for older transformers versions
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if not hasattr(config, "_attn_implementation"):
            config._attn_implementation = "eager"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        
        # Patch for older transformers versions where this method is missing but called by remote code
        if not hasattr(model, "warn_if_padding_and_no_attention_mask"):
            model.warn_if_padding_and_no_attention_mask = lambda *args, **kwargs: None
            
        model.eval()
        model.to(device)

        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        loaded_gte_models[model_name] = (tokenizer, model)

    if isinstance(text, str):
        text = [text]

    embeddings = []
    for t in text:
        # Use raw text for speed optimization (GTE can work without instructions)
        inputs = tokenizer(t, return_tensors="pt", truncation=True,
                          padding=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # GTE models use the last token's representation as the embedding
            emb = outputs.last_hidden_state[:, 0]  # [1, hidden_size]
            embeddings.append(emb.cpu())

    if embeddings:
        result = torch.cat(embeddings, dim=0)  # [num_texts, hidden_size]
    else:
        # Fallback for empty input - GTE-Qwen2-1.5B has 1536 hidden size
        result = torch.empty(0, 1536)

    return result
