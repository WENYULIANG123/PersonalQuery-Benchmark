from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
import os

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





def call_llm_with_retry(llm_model, prompt: str, max_retries: int = 3, context: str = "unknown") -> tuple[str, bool]:
    """Call LLM without timeout restrictions"""
    # Track API key usage
    api_info = get_current_api_info()

    for attempt in range(max_retries):
        try:
            messages = [{"role": "user", "content": prompt}]
            response = llm_model.invoke(messages)

            # Successful response
            if response and hasattr(response, 'content') and response.content is not None:
                return str(response.content).strip(), True
            else:
                print(f"LLM error [{api_info}]: Invalid response format (attempt {attempt + 1}/{max_retries})", flush=True)
                if attempt < max_retries - 1:
                    continue
                raise APIErrorException(f"Invalid response format with {api_info}")

        except Exception as e:
            # API call failed with an exception
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"LLM error [{api_info}]: {error_msg} (attempt {attempt + 1}/{max_retries})", flush=True)

            # If it's an authentication error, don't retry
            if ("401" in error_msg or
                "authentication" in error_msg.lower() or
                "api key" in error_msg.lower() or
                "AuthenticationError" in error_type or
                "invalid" in error_msg.lower()):
                print("❌ Authentication failed - API key appears to be invalid", flush=True)
                print("💡 请检查您的硅基流动API Key是否正确", flush=True)
                raise APIErrorException(f"Authentication error with {api_info}: {error_msg}")

            # For other errors, retry if attempts remain
            if attempt < max_retries - 1:
                continue
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
        ApiProvider.SILICONFLOW: ('SILICONFLOW_MODEL', 'Qwen/Qwen3-8B'),
    }

    if provider in model_configs:
        env_var, default_model = model_configs[provider]
        model_env = os.getenv(env_var)
        if model_env:
            return model_env
        return default_model

    return None

def _create_chat_model(temperature: float = 0.7, max_tokens: int = 500, timeout: int = None, scope: str = "default") -> BaseChatModel:
    """Create a ChatOpenAI model with the specified parameters."""
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

    # Only add timeout if it's not None
    if timeout is not None:
        chatopenai_kwargs['timeout'] = timeout

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
