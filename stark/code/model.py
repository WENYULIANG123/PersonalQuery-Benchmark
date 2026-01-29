from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from openai import OpenAI
import os
import json
import threading
import time
import tempfile
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

# API Provider type
class ApiProvider:
    SILICONFLOW = 'siliconflow'

class APIErrorException(Exception):
    """Exception raised when any API error occurs to trigger API key switch."""
    pass

# Simplified API Keys & Configuration
SILICONFLOW_API_KEY = 'sk-drezmfyckjkmxixpiblvbwdhypjbrsoyvmeertajtupiqnnj'
DEFAULT_MODEL = 'THUDM/GLM-Z1-9B-0414'
SILICONFLOW_BASE_URL = 'https://api.siliconflow.cn/v1'

_api_response_lock = threading.Lock()
_api_responses_file: Optional[str] = None

def set_api_responses_file(file_path: str, overwrite: bool = False):
    """Set the file path for saving API raw responses."""
    global _api_responses_file
    _api_responses_file = file_path
    if overwrite and os.path.exists(file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([], f)
        except Exception as e:
            print(f"⚠️ Failed to clear API response file: {e}", flush=True)

def _resolve_thinking_budget(llm_model, default: int = 32768) -> Optional[int]:
    """Resolve thinking budget from env or model config."""
    env = os.getenv("SILICONFLOW_THINKING_BUDGET")
    if env is not None:
        v = env.strip().lower()
        if v in {"", "none", "null", "false", "off", "0"}: return None
        try:
            n = int(v)
            return None if n <= 0 else n
        except: pass

    for attr in ("extra_body", "model_kwargs", "kwargs"):
        d = getattr(llm_model, attr, None)
        if isinstance(d, dict):
            tb = d.get("thinking_budget")
            try:
                n = int(tb)
                return None if n <= 0 else n
            except: continue
    return default

def _save_api_response(context: str, prompt: str, response_dict: dict, api_info: str, success: bool, error: str = None, meta: dict = None):
    """Save API raw response in a thread-safe manner."""
    global _api_responses_file
    if not _api_responses_file: return
    
    try:
        def _compact_prompt(p: str, max_len: int = 3000) -> str:
            if not p or len(p) <= max_len: return p
            text_marker = "Text:"
            idx = p.find(text_marker)
            if idx != -1:
                head = p[: min(400, idx)]
                tail = p[idx : idx + (max_len - len(head) - 32)]
                return f"{head}\n...[TRUNCATED]...\n{tail}"
            return f"{p[:max_len//2]}\n...[TRUNCATED]...\n{p[-(max_len//2-32):]}"

        compact_prompt = _compact_prompt(prompt)
        content = response_dict.get('content', '')
        reasoning = response_dict.get('reasoning_content', '')

        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "context": context, "api_info": api_info, "success": success,
            "prompt": compact_prompt, "prompt_length": len(prompt),
            "meta": meta or {},
            "raw_response": {"reasoning_content": reasoning, "content": content},
            "response_length": len(content), "reasoning_length": len(reasoning),
            "error": error if not success else None
        }
        
        with _api_response_lock:
            history = []
            if os.path.exists(_api_responses_file):
                try:
                    with open(_api_responses_file, 'r', encoding='utf-8') as f: history = json.load(f)
                except: pass
            history.append(data)
            with open(_api_responses_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Failed to save API response: {e}", flush=True)

def _extract_content_and_reasoning(response: Any) -> Tuple[str, str]:
    """Unified extractor for reasoning_content and content from OpenAI or LangChain responses."""
    content = ""
    reasoning = ""

    # Try OpenAI-style (Choice/Message object)
    if hasattr(response, 'choices') and len(response.choices) > 0:
        msg = response.choices[0].message
        content = getattr(msg, 'content', '') or ''
        reasoning = (getattr(msg, 'reasoning_content', None) or 
                     getattr(msg, 'reasoning', None) or 
                     (msg.get('reasoning_content') if isinstance(msg, dict) else ''))
    # Try LangChain-style (AIMessage)
    elif hasattr(response, 'content'):
        content = response.content
        reasoning = getattr(response, 'reasoning_content', None)
        if not reasoning and hasattr(response, 'response_metadata'):
            meta = response.response_metadata
            reasoning = meta.get('reasoning_content') or meta.get('reasoning')
        if not reasoning and hasattr(response, 'additional_kwargs'):
            kwargs = response.additional_kwargs
            reasoning = kwargs.get('reasoning_content') or kwargs.get('reasoning')

    return str(content or "").strip(), str(reasoning or "").strip()

def call_llm_with_retry(llm_model, prompt: str, max_retries: int = 3, context: str = "unknown", use_openai_client: bool = True, meta: Optional[dict] = None) -> Tuple[str, bool]:
    api_info = get_current_api_info()
    
    for attempt in range(max_retries):
        try:
            res_dict = {"content": "", "reasoning_content": ""}
            if use_openai_client:
                client = OpenAI(base_url=SILICONFLOW_BASE_URL, api_key=SILICONFLOW_API_KEY)
                resp = client.chat.completions.create(
                    model=get_model_name(ApiProvider.SILICONFLOW),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=getattr(llm_model, 'max_tokens', 500),
                    temperature=getattr(llm_model, 'temperature', 0.7),
                    extra_body={"thinking_budget": _resolve_thinking_budget(llm_model)} if _resolve_thinking_budget(llm_model) else None
                )
                content, reasoning = _extract_content_and_reasoning(resp)
            else:
                resp = llm_model.invoke([{"role": "user", "content": prompt}])
                content, reasoning = _extract_content_and_reasoning(resp)

            res_dict = {"content": content, "reasoning_content": reasoning}
            if content:
                _save_api_response(context, prompt, res_dict, api_info, True, meta=meta)
                return content, True
            
            raise Exception("Empty content in response")

        except Exception as e:
            err_msg = str(e)
            print(f"LLM error [{api_info}]: {err_msg} (attempt {attempt+1}/{max_retries})", flush=True)
            if any(x in err_msg.lower() for x in ["401", "authentication", "api key", "invalid"]):
                _save_api_response(context, prompt, res_dict, api_info, False, err_msg, meta=meta)
                raise APIErrorException(f"Auth error: {err_msg}")
            
            if attempt == max_retries - 1:
                _save_api_response(context, prompt, res_dict, api_info, False, err_msg, meta=meta)
                raise APIErrorException(f"API error: {err_msg}")

def get_all_api_keys_in_order() -> List[Dict[str, Any]]:
    return [{
        'provider': ApiProvider.SILICONFLOW, 'api_key': SILICONFLOW_API_KEY,
        'key_index': 0, 'provider_name': 'SiliconFlow', 'key_id': 'SiliconFlow-Key#1'
    }]

def get_api_provider():
    return {'provider': ApiProvider.SILICONFLOW, 'api_key': SILICONFLOW_API_KEY, 'key_index': 0}

def get_current_api_info():
    return "SiliconFlow-Key#1"

def get_model_name(provider=ApiProvider.SILICONFLOW):
    return os.getenv('SILICONFLOW_MODEL', DEFAULT_MODEL)

def get_siliconflow_config():
    return {'base_url': os.getenv('SILICONFLOW_BASE_URL', SILICONFLOW_BASE_URL), 'default_headers': {}}

def _create_chat_model(temperature: float = 0.7, max_tokens: int = 500, scope: str = "default", thinking_budget: int = 32768) -> BaseChatModel:
    os.environ['OPENAI_BASE_URL'] = SILICONFLOW_BASE_URL
    return ChatOpenAI(
        model=get_model_name(), temperature=temperature, max_tokens=max_tokens,
        api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL,
        extra_body={'thinking_budget': thinking_budget} if thinking_budget else None
    )

def get_router_model(): return _create_chat_model(temperature=0, max_tokens=50, scope="router")
def get_gm_model(): return _create_chat_model(temperature=0.1, max_tokens=2000, scope="general")

# --- Batch Inference Interface ---

def submit_batch_inference(prompts: List[str], model: str = "Qwen/QwQ-32B", max_tokens: int = 4096, thinking_budget: int = 32768, custom_ids: Optional[List[str]] = None) -> str:
    """
    Submits a batch inference job to SiliconFlow.
    Returns the batch job ID.
    """
    client = OpenAI(base_url=SILICONFLOW_BASE_URL, api_key=SILICONFLOW_API_KEY)
    
    # Create a temporary JSONL file for the batch input
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as tmp:
        for i, prompt in enumerate(prompts):
            cid = custom_ids[i] if custom_ids and i < len(custom_ids) else f"req-{i}"
            request = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "thinking_budget": thinking_budget
                }
            }
            tmp.write(json.dumps(request, ensure_ascii=False) + '\n')
        tmp_path = tmp.name

    try:
        # 1. Upload the file
        with open(tmp_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        
        # Accessing ID safely as some SDK versions/wrappers might differ
        file_id = getattr(batch_input_file, 'id', None)
        if not file_id and hasattr(batch_input_file, 'data'):
            file_id = batch_input_file.data.get('id')
        
        if not file_id:
            raise Exception(f"Failed to get file ID from upload response: {batch_input_file}")

        # 2. Create the batch job
        batch = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            extra_body={"replace": {"model": model}}
        )
        
        batch_id = getattr(batch, 'id', None)
        if not batch_id and hasattr(batch, 'data'):
            batch_id = batch.data.get('id')
        
        # Log batch submission
        _save_api_response(
            context="batch_submission",
            prompt=f"Submitted batch with {len(prompts)} prompts. Model: {model}",
            response_dict={"content": f"Batch ID: {batch_id}", "batch_id": batch_id},
            api_info="SiliconFlow-Batch-Submit",
            success=True,
            meta={"prompt_count": len(prompts), "model": model, "batch_id": batch_id}
        )
            
        return batch_id
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def get_batch_status(batch_id: str) -> Dict[str, Any]:
    """Retrieves the full status object of a batch job."""
    client = OpenAI(base_url=SILICONFLOW_BASE_URL, api_key=SILICONFLOW_API_KEY)
    batch = client.batches.retrieve(batch_id)
    
    # Convert To Dict if it's an object
    if hasattr(batch, 'model_dump'):
        return batch.model_dump()
    elif hasattr(batch, 'to_dict'):
        return batch.to_dict()
    return str(batch) # Fallback

def retrieve_batch_results(batch_id: str) -> List[Dict[str, Any]]:
    """
    Downloads and parses the results of a completed batch job.
    Returns a list of result dictionaries.
    """
    import requests
    client = OpenAI(base_url=SILICONFLOW_BASE_URL, api_key=SILICONFLOW_API_KEY)
    batch = client.batches.retrieve(batch_id)
    
    status = getattr(batch, 'status', None)
    if status != "completed":
        print(f"Batch {batch_id} is not completed (status: {status})")
        return []
    
    output_file_id = getattr(batch, 'output_file_id', None)
    if not output_file_id:
        print(f"No output file found for batch {batch_id}")
        return []
        
    # Use direct requests. SiliconFlow output_file_id can be a literal ID or a full URL.
    headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}"}
    if output_file_id.startswith("http"):
        url = output_file_id
    else:
        url = f"{SILICONFLOW_BASE_URL}/files/{output_file_id}/content"
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers if not output_file_id.startswith("http") else None, timeout=60)
            response.raise_for_status()
            content = response.text
            break # Success
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to download batch results after {max_retries} attempts: {e}")
                return []
            print(f"Attempt {attempt+1} failed to download batch results: {e}. Retrying...")
            time.sleep(2)
    
    results = []
    for line in content.strip().split('\n'):
        if line.strip():
            try:
                res_item = json.loads(line)
                results.append(res_item)
                
                # Log individual batch result
                # Extract prompt and response info for logging
                res_body = res_item.get('response', {}).get('body', {})
                res_content = ""
                res_reasoning = ""
                if 'choices' in res_body and res_body['choices']:
                    msg = res_body['choices'][0].get('message', {})
                    res_content = msg.get('content', '')
                    res_reasoning = msg.get('reasoning_content', '')
                
                _save_api_response(
                    context=f"batch_result_item",
                    prompt=f"Batch Result for ID: {res_item.get('id')}, Custom ID: {res_item.get('custom_id')}",
                    response_dict={"content": res_content, "reasoning_content": res_reasoning},
                    api_info="SiliconFlow-Batch-Result",
                    success=True,
                    meta={
                        "batch_id": batch_id, 
                        "custom_id": res_item.get('custom_id'),
                        "request_id": res_item.get('id')
                    }
                )
            except:
                continue
    return results

async def wait_for_batch_results(batch_id: str, poll_interval: int = 60) -> List[Dict[str, Any]]:
    """Polls for batch completion and returns results."""
    import asyncio
    while True:
        batch = get_batch_status(batch_id)
        status = batch.get('status') if isinstance(batch, dict) else getattr(batch, 'status', None)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {batch_id} status: {status}", flush=True)
        
        if status == "completed":
            return retrieve_batch_results(batch_id)
        if status in ["failed", "expired", "cancelled"]:
            print(f"Batch {batch_id} terminated with status: {status}")
            return []
            
        await asyncio.sleep(poll_interval)

if __name__ == "__main__":
    import sys
    
    # Check if we want to test batch
    if "--batch" in sys.argv:
        print("🚀 Testing Batch Inference...")
        try:
            prompts = ["Explain quantum entanglement in one sentence.", "What is the capital of France?"]
            batch_id = submit_batch_inference(prompts)
            print(f"Batch submitted! ID: {batch_id}")
            print("Waiting for results (this may take a while)...")
            results = wait_for_batch_results(batch_id, poll_interval=10)
            print(f"Results: {json.dumps(results, indent=2, ensure_ascii=False)}")
        except Exception as e:
            print(f"Batch Error: {e}")
    else:
        print("🚀 Testing LLM Call (use --batch to test batch inference)...")
        try:
            model = get_router_model()
            content, ok = call_llm_with_retry(model, "Hello, say 'test successful'", max_retries=1)
            print(f"Result: {content}" if ok else "Failed")
        except Exception as e:
            print(f"Error: {e}")
