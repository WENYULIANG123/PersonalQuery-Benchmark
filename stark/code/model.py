from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from openai import OpenAI
import os
import json
import threading
from datetime import datetime
from typing import Optional

# API Provider type
class ApiProvider:
    CEREBRAS = 'cerebras'
    SILICONFLOW = 'siliconflow'  # Keep for fallback

class APIErrorException(Exception):
    """Exception raised when any API error occurs to trigger API key switch."""
    pass

# API Keys
SILICONFLOW_API_KEY = 'sk-drezmfyckjkmxixpiblvbwdhypjbrsoyvmeertajtupiqnnj'

# API key management - single key configuration

# Global lock for thread-safe file writing
_api_response_lock = threading.Lock()
_api_responses_file: Optional[str] = None

def set_api_responses_file(file_path: str):
    """Set the file path for saving API raw responses."""
    global _api_responses_file
    _api_responses_file = file_path

def _resolve_thinking_budget(llm_model, default: Optional[int] = 32768) -> Optional[int]:
    """
    Resolve SiliconFlow/DeepSeek-R1 thinking budget.

    Priority:
    1) env var SILICONFLOW_THINKING_BUDGET (supports: unset -> fallback; "none"/"null"/"0" -> None)
    2) llm_model.extra_body / llm_model.model_kwargs / llm_model.kwargs (if present)
    3) default
    """
    env = os.getenv("SILICONFLOW_THINKING_BUDGET")
    if env is not None:
        v = env.strip().lower()
        if v in {"", "none", "null", "false", "off", "0"}:
            return None
        try:
            n = int(v)
            return None if n <= 0 else n
        except Exception:
            # If malformed, fall back to other sources/default
            pass

    # Try to get thinking_budget from common LangChain storage locations
    for attr in ("extra_body", "model_kwargs", "kwargs"):
        d = getattr(llm_model, attr, None)
        if isinstance(d, dict):
            tb = d.get("thinking_budget")
            if tb is None:
                continue
            try:
                n = int(tb)
                return None if n <= 0 else n
            except Exception:
                continue

    return default

def _save_api_response(
    context: str,
    prompt: str,
    raw_response: dict,
    api_info: str,
    success: bool,
    error: Optional[str] = None,
    meta: Optional[dict] = None,
):
    """Save API raw response to JSON file in a thread-safe manner.
    
    Args:
        raw_response: Dictionary containing 'reasoning_content' and 'content' fields
    """
    global _api_responses_file
    if not _api_responses_file:
        return
    
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Extract reasoning_content and content from raw_response dict
        reasoning_content = raw_response.get('reasoning_content', '')
        content = raw_response.get('content', '')
        
        # NOTE:
        # We previously truncated prompts to the first 1000 chars. For prompts where the
        # actual data is appended at the end (e.g., "... Text: <review>"), this removes
        # the review text and breaks downstream matching logic that searches the prompt.
        # Keep a compact but useful slice that preserves the "Text:" section if present.
        def _compact_prompt(p: str, max_len: int = 3000) -> str:
            if not p:
                return ""
            if len(p) <= max_len:
                return p
            text_marker = "Text:"
            idx = p.find(text_marker)
            if idx != -1:
                # Keep a small header for context + the beginning of the Text section
                head = p[: min(400, idx)]
                tail = p[idx : idx + (max_len - len(head) - 32)]
                return f"{head}\n...[TRUNCATED]...\n{tail}"
            # Fallback: keep both head and tail
            head = p[: max_len // 2]
            tail = p[-(max_len - len(head) - 32) :]
            return f"{head}\n...[TRUNCATED]...\n{tail}"

        compact_prompt = _compact_prompt(prompt, max_len=3000)

        response_data = {
            "timestamp": timestamp,
            "context": context,
            "api_info": api_info,
            "success": success,
            "prompt": compact_prompt,
            "prompt_length": len(prompt),
            "prompt_stored_length": len(compact_prompt),
            "meta": meta or {},
            "raw_response": {
                "reasoning_content": reasoning_content,
                "content": content
            },
            "response_length": len(content) if content else 0,
            "reasoning_length": len(reasoning_content) if reasoning_content else 0,
            "error": error if not success else None
        }
        
        with _api_response_lock:
            # Read existing data if file exists
            existing_data = []
            if os.path.exists(_api_responses_file):
                try:
                    with open(_api_responses_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except Exception:
                    existing_data = []
            
            # Append new response
            existing_data.append(response_data)
            
            # Write back to file
            with open(_api_responses_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
    except Exception as e:
        # Don't fail the main operation if saving fails
        print(f"⚠️ Failed to save API response to file: {e}", flush=True)

def _call_llm_with_openai_client(prompt: str, model_name: str, base_url: str, api_key: str, 
                                  temperature: float = 0.7, max_tokens: int = 500, 
                                  timeout: Optional[int] = None, thinking_budget: int = None) -> tuple[dict, Optional[str]]:
    """Call LLM using OpenAI client directly for better reasoning_content extraction.
    
    Returns:
        tuple: (response_dict with 'reasoning_content' and 'content', error_message)
    """
    try:
        client_kwargs = {
            "base_url": base_url,
            "api_key": api_key,
        }
        # If timeout is explicitly provided, respect it; otherwise let client use its own defaults
        if timeout is not None:
            client_kwargs["timeout"] = timeout / 1000.0

        client = OpenAI(**client_kwargs)
        
        messages = [{"role": "user", "content": prompt}]
        
        # Build request parameters
        request_params = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add thinking_budget via extra_body if provided
        if thinking_budget is not None:
            request_params["extra_body"] = {
                "thinking_budget": thinking_budget
            }
        
        response = client.chat.completions.create(**request_params)
        
        # Extract content and reasoning_content directly from response
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message
            
            content = message.content if message.content else ""
            reasoning_content = ""
            
            # Debug: Print all available attributes to understand response structure
            # print(f"🔍 Debug - message attributes: {dir(message)}", flush=True)
            # print(f"🔍 Debug - choice attributes: {dir(choice)}", flush=True)
            # print(f"🔍 Debug - response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}", flush=True)
            
            # Extract reasoning_content if available - try multiple methods
            # Method 1: Direct attribute
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning_content = message.reasoning_content
                #print(f"✅ Found reasoning_content in message.reasoning_content (length: {len(reasoning_content)})", flush=True)
            # Method 2: Alternative attribute name
            elif hasattr(message, 'reasoning') and message.reasoning:
                reasoning_content = message.reasoning
                #print(f"✅ Found reasoning_content in message.reasoning (length: {len(reasoning_content)})", flush=True)
            # Method 3: Check if it's in the raw response object
            elif hasattr(choice, 'reasoning_content') and choice.reasoning_content:
                reasoning_content = choice.reasoning_content
                #print(f"✅ Found reasoning_content in choice.reasoning_content (length: {len(reasoning_content)})", flush=True)
            # Method 4: Check response object itself
            elif hasattr(response, 'reasoning_content') and response.reasoning_content:
                reasoning_content = response.reasoning_content
                #print(f"✅ Found reasoning_content in response.reasoning_content (length: {len(reasoning_content)})", flush=True)
            # Method 5: Try to get from message dict if it's a dict-like object
            elif isinstance(message, dict) and 'reasoning_content' in message:
                reasoning_content = message['reasoning_content']
                #print(f"✅ Found reasoning_content in message dict (length: {len(reasoning_content)})", flush=True)
            else:
                print(f"⚠️ No reasoning_content found in response (thinking_budget={thinking_budget})", flush=True)
            
            return {
                "reasoning_content": reasoning_content.strip() if reasoning_content else "",
                "content": content.strip() if content else ""
            }, None
        else:
            return {"reasoning_content": "", "content": ""}, "No choices in response"
            
    except Exception as e:
        return {"reasoning_content": "", "content": ""}, str(e)


def call_llm_with_retry(
    llm_model,
    prompt: str,
    max_retries: int = 3,
    context: str = "unknown",
    use_openai_client: bool = True,
    meta: Optional[dict] = None,
) -> tuple[str, bool]:
    """Call LLM without timeout restrictions.
    
    Args:
        llm_model: LangChain model instance (used for config if use_openai_client=True)
        prompt: The prompt to send
        max_retries: Maximum number of retry attempts
        context: Context identifier for logging
        use_openai_client: If True, use OpenAI client directly for better reasoning_content extraction
    """
    # Track API key usage
    api_info = get_current_api_info()
    error_msg = None

    # If using OpenAI client directly, extract config from model
    if use_openai_client:
        try:
            result = get_api_provider()
            if result['provider'] == ApiProvider.SILICONFLOW:
                config = get_siliconflow_config()
                model_name = get_model_name(ApiProvider.SILICONFLOW)
                api_key = result['api_key']
                base_url = config['base_url']
                
                # Try to get parameters from model if available
                temperature = getattr(llm_model, 'temperature', 0.7)
                max_tokens = getattr(llm_model, 'max_tokens', 500)
                # Do not enforce a client-side timeout here; let the provider handle it.
                timeout = None
                # IMPORTANT:
                # Do NOT hardcode a tiny thinking_budget (it truncates reasoning_content).
                # Use env/model config; default is large enough for full reasoning.
                thinking_budget = _resolve_thinking_budget(llm_model, default=32768)
                
                for attempt in range(max_retries):
                    raw_response_dict, error = _call_llm_with_openai_client(
                        prompt, model_name, base_url, api_key,
                        temperature, max_tokens, timeout, thinking_budget
                    )
                    
                    if error is None:
                        # Success
                        content = raw_response_dict.get('content', '')
                        if content:
                            _save_api_response(context, prompt, raw_response_dict, api_info, True, meta=meta)
                            return content, True
                        else:
                            error_msg = f"Empty content in response (attempt {attempt + 1}/{max_retries})"
                            print(f"LLM error [{api_info}]: {error_msg}", flush=True)
                            if attempt < max_retries - 1:
                                continue
                            _save_api_response(context, prompt, raw_response_dict, api_info, False, error_msg, meta=meta)
                            raise APIErrorException(f"Empty content with {api_info}")
                    else:
                        error_msg = error
                        error_type = type(error).__name__ if hasattr(error, '__class__') else 'Exception'
                        print(f"LLM error [{api_info}]: {error_msg} (attempt {attempt + 1}/{max_retries})", flush=True)
                        
                        # If it's an authentication error, don't retry
                        if ("401" in error_msg or
                            "authentication" in error_msg.lower() or
                            "api key" in error_msg.lower() or
                            "AuthenticationError" in error_type or
                            "invalid" in error_msg.lower()):
                            print("❌ Authentication failed - API key appears to be invalid", flush=True)
                            print("💡 请检查您的硅基流动API Key是否正确", flush=True)
                            _save_api_response(context, prompt, raw_response_dict, api_info, False, error_msg, meta=meta)
                            raise APIErrorException(f"Authentication error with {api_info}: {error_msg}")
                        
                        # For other errors, retry if attempts remain
                        if attempt < max_retries - 1:
                            continue
                        _save_api_response(context, prompt, raw_response_dict, api_info, False, error_msg, meta=meta)
                        raise APIErrorException(f"API error with {api_info}: {error_msg}")
            else:
                # Fall back to LangChain if not SiliconFlow
                use_openai_client = False
        except Exception as e:
            # If OpenAI client approach fails, fall back to LangChain
            print(f"⚠️ OpenAI client approach failed, falling back to LangChain: {e}", flush=True)
            use_openai_client = False

    # Fallback to LangChain approach
    if not use_openai_client:
        for attempt in range(max_retries):
            try:
                messages = [{"role": "user", "content": prompt}]
                response = llm_model.invoke(messages)

                # Successful response
                if response and hasattr(response, "content") and response.content is not None:
                    # Extract reasoning_content and content from response
                    reasoning_content = ""
                    content = str(response.content).strip()

                    # Try multiple ways to extract reasoning_content
                    # Method 1: Check if response has reasoning_content attribute directly
                    if hasattr(response, "reasoning_content") and response.reasoning_content:
                        reasoning_content = str(response.reasoning_content).strip()
                    # Method 2: Check response_metadata
                    elif hasattr(response, "response_metadata") and response.response_metadata:
                        metadata = response.response_metadata
                        if isinstance(metadata, dict):
                            if "reasoning_content" in metadata:
                                reasoning_content = str(metadata["reasoning_content"]).strip()
                            # Check for other possible keys
                            elif "reasoning" in metadata:
                                reasoning_content = str(metadata["reasoning"]).strip()
                    # Method 3: Check if response has additional_kwargs (for LangChain compatibility)
                    elif hasattr(response, "additional_kwargs") and response.additional_kwargs:
                        additional = response.additional_kwargs
                        if isinstance(additional, dict):
                            if "reasoning_content" in additional:
                                reasoning_content = str(additional["reasoning_content"]).strip()
                            elif "reasoning" in additional:
                                reasoning_content = str(additional["reasoning"]).strip()
                    # Method 4: Try to access raw response if available
                    elif hasattr(response, "response") and hasattr(response.response, "choices"):
                        # Try to extract from raw API response choices
                        try:
                            choices = response.response.choices
                            if choices and len(choices) > 0:
                                choice = choices[0]
                                if hasattr(choice, "message"):
                                    message = choice.message
                                    if hasattr(message, "reasoning_content"):
                                        reasoning_content = str(message.reasoning_content).strip()
                                    elif hasattr(message, "reasoning"):
                                        reasoning_content = str(message.reasoning).strip()
                        except Exception:
                            pass

                    # Prepare raw_response dict with both fields
                    raw_response_dict = {
                        "reasoning_content": reasoning_content,
                        "content": content,
                    }

                    # Save successful response
                    _save_api_response(context, prompt, raw_response_dict, api_info, True, meta=meta)
                    return content, True

                error_msg = f"Invalid response format (attempt {attempt + 1}/{max_retries})"
                print(f"LLM error [{api_info}]: {error_msg}", flush=True)
                if attempt < max_retries - 1:
                    continue
                # Save failed response
                _save_api_response(
                    context,
                    prompt,
                    {"reasoning_content": "", "content": ""},
                    api_info,
                    False,
                    error_msg,
                    meta=meta,
                )
                raise APIErrorException(f"Invalid response format with {api_info}")

            except Exception as e:
                # API call failed with an exception
                error_msg = str(e)
                error_type = type(e).__name__
                print(f"LLM error [{api_info}]: {error_msg} (attempt {attempt + 1}/{max_retries})", flush=True)

                # If it's an authentication error, don't retry
                if (
                    "401" in error_msg
                    or "authentication" in error_msg.lower()
                    or "api key" in error_msg.lower()
                    or "AuthenticationError" in error_type
                    or "invalid" in error_msg.lower()
                ):
                    print("❌ Authentication failed - API key appears to be invalid", flush=True)
                    print("💡 请检查您的硅基流动API Key是否正确", flush=True)
                    _save_api_response(
                        context,
                        prompt,
                        {"reasoning_content": "", "content": ""},
                        api_info,
                        False,
                        error_msg,
                        meta=meta,
                    )
                    raise APIErrorException(f"Authentication error with {api_info}: {error_msg}")

                # For other errors, retry if attempts remain
                if attempt < max_retries - 1:
                    continue
                # Save failed response on final attempt
                _save_api_response(
                    context,
                    prompt,
                    {"reasoning_content": "", "content": ""},
                    api_info,
                    False,
                    error_msg,
                    meta=meta,
                )
                raise APIErrorException(f"API error with {api_info}: {error_msg}")

# Helper function to collect all API keys
def _collect_all_api_keys():
    """Collect all available API keys in order of preference."""
    all_api_keys = []
    # Only use SiliconFlow
    all_api_keys.append((ApiProvider.SILICONFLOW, 0, SILICONFLOW_API_KEY))

    return all_api_keys

# Helper function to collect all API keys with full config
def _collect_all_api_keys_with_config():
    """Collect all available API keys with full configuration."""
    all_api_keys = []
    # Only use SiliconFlow
    all_api_keys.append({
        'provider': ApiProvider.SILICONFLOW,
        'api_key': SILICONFLOW_API_KEY,
        'key_index': 0,
        'provider_name': 'SiliconFlow',
        'key_id': 'SiliconFlow-Key#1'
    })
    return all_api_keys

# Public function to get all API keys in order (for main script usage)
def get_all_api_keys_in_order():
    """Get all available API keys in order of preference for sequential usage."""
    return _collect_all_api_keys_with_config()

# Get API provider and key - Simplified (main script handles sequential usage)
def get_api_provider():
    all_api_keys = _collect_all_api_keys()

    if not all_api_keys:
        print('⚠️  No API keys configured. Using mock configuration for testing')
        return {'provider': 'mock', 'api_key': 'mock_key'}

    # Default to first available key (main script will override this)
    provider, key_index, selected_key = all_api_keys[0]
    return {'provider': provider, 'api_key': selected_key, 'key_index': key_index}





# Helper function to get provider display info
def _get_provider_display_info(result):
    """Get provider display name and key info from API provider result."""
    provider_name = result['provider']
    if provider_name == ApiProvider.SILICONFLOW:
        display_name = "SiliconFlow"
        key_index = result.get('key_index', 0)
        key_info = f"Key #{key_index + 1}"
        return display_name, key_info
    else:
        return provider_name, ""

# Get current API provider info for error reporting
def get_current_api_info():
    """Returns information about the currently active API key."""
    try:
        result = get_api_provider()
        display_name, key_info = _get_provider_display_info(result)
        if key_info:
            return f"{display_name}-{key_info.replace(' ', '')}"
        else:
            return display_name
    except Exception:
        return "Unknown"

# Get base URL and headers for provider (env overrides allowed)
def get_provider_config(provider):
    """Get configuration for the specified provider."""
    configs = {
        ApiProvider.SILICONFLOW: {
            'env_var': 'SILICONFLOW_BASE_URL',
            'default_url': 'https://api.siliconflow.cn/v1'
        }
    }

    if provider not in configs:
        raise ValueError(f"Unsupported provider: {provider}")

    config = configs[provider]
    env_base = os.getenv(config['env_var'], config['default_url'])
    return {
        'base_url': env_base,
        'default_headers': {},  # All providers don't need special headers currently
    }

# Backward compatibility functions

def get_siliconflow_config():
    return get_provider_config(ApiProvider.SILICONFLOW)


def log_model_selection(_scope, _model_name, _base_url=None, _provider=None, _headers=None):
    # no-op: logging removed for production
    pass

# Get model name for provider (env overrides allowed)
def get_model_name(provider):
    """Get model name for provider with environment variable override."""
    model_configs = {
        ApiProvider.SILICONFLOW: ('SILICONFLOW_MODEL', 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B'),
    }

    if provider in model_configs:
        env_var, default_model = model_configs[provider]
        model_env = os.getenv(env_var)
        if model_env:
            return model_env
        return default_model

    return None

def _create_chat_model(temperature: float = 0.7, max_tokens: int = 500, timeout: int = 60000, scope: str = "default", thinking_budget: int = 32768) -> BaseChatModel:
    """Create a ChatOpenAI model with the specified parameters.
    
    Args:
        temperature: Sampling temperature for the model
        max_tokens: Maximum number of tokens to generate
        timeout: Request timeout in milliseconds (default: 60s)
        scope: Scope identifier for logging
        thinking_budget: Maximum thinking budget tokens (default: 32768). Set <=0 to disable.
    """
    result = get_api_provider()

    if result['provider'] == 'mock':
        raise ValueError('Mock provider - use InteractionManager fallback logic instead')

    # Single API key configuration - no tracking needed

    # Get provider config and model name
    if result['provider'] == ApiProvider.SILICONFLOW:
        config = get_siliconflow_config()
        model_name = get_model_name(ApiProvider.SILICONFLOW)
    else:
        raise ValueError(f"Unsupported API provider: {result['provider']}")

    log_model_selection(scope, model_name, config['base_url'], result['provider'], config['default_headers'])
    os.environ['OPENAI_BASE_URL'] = config['base_url']

    # Build kwargs for ChatOpenAI initialization
    chatopenai_kwargs = {
        'model': model_name,
        'temperature': temperature,
        'api_key': result['api_key'],
        'base_url': config['base_url'],
        'default_headers': config['default_headers'],
        'max_tokens': max_tokens,
    }

    # Add thinking_budget via extra_body for SiliconFlow API
    # This is a custom parameter for SiliconFlow, not a standard OpenAI parameter
    if thinking_budget is not None:
        chatopenai_kwargs['extra_body'] = {
            'thinking_budget': thinking_budget
        }

    return ChatOpenAI(**chatopenai_kwargs)



def get_router_model() -> BaseChatModel:
    """
    Returns the LLM used for routing intents.
    Needs to be fast and structured.
    """
    return _create_chat_model(temperature=0, max_tokens=50, timeout=30000, scope="router")


def get_gm_model() -> BaseChatModel:
    """
    Returns the general LLM model for analysis tasks.
    Suitable for longer, more complex responses.
    """
    return _create_chat_model(temperature=0.1, max_tokens=2000, timeout=60000, scope="general")



def test_llm_call():
    """测试LLM调用的主函数"""
    print("🚀 测试LLM调用功能")
    print("=" * 50)

    # 检查API配置
    result = get_api_provider()
    provider_display, key_info = _get_provider_display_info(result)

    print(f"📊 API Provider: {provider_display} {key_info}")
    print(f"🔑 API Key: {'已配置' if result['api_key'] != 'mock_key' else 'Mock模式'}")

    if result['provider'] == 'mock':
        print("⚠️ 使用Mock模式 - 将返回预设响应")
        return

    try:
        # 初始化路由模型（轻量级，适合测试）
        print("\n🤖 初始化路由模型...")
        router_model = get_router_model()
        print(f"✅ 模型类型: {type(router_model).__name__}")

        # 测试简单的查询
        test_query = "Hello, can you respond with 'LLM test successful'?"
        print(f"\n📤 发送测试查询: {test_query}")

        # 调用LLM (使用改进的错误处理)
        response_content, success = call_llm_with_retry(router_model, test_query, max_retries=1, context="test")

        if not success:
            print("❌ LLM调用失败")
            return False

        print(f"📝 响应长度: {len(response_content)} 字符")
        print(f"📄 响应内容: {response_content[:200]}{'...' if len(response_content) > 200 else ''}")

        # 验证响应
        if "successful" in response_content.lower() or "test" in response_content.lower():
            print("✅ LLM响应正常！")
            return True
        else:
            print("⚠️ LLM响应内容可能异常")
            return False

    except Exception as e:
        print(f"❌ LLM调用失败: {e}")
        print(f"❌ 错误类型: {type(e).__name__}")
        return False



if __name__ == "__main__":
    success = test_llm_call()
    exit(0 if success else 1)
