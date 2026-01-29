#!/usr/bin/env python3
"""
共享工具函数模块
包含各个模块都需要使用的通用函数
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log_with_timestamp(message: str):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def get_all_api_keys_in_order():
    """Get all API keys in order - wrapper for model function."""
    from model import get_all_api_keys_in_order as model_get_keys
    return model_get_keys()


def create_llm_with_config(api_config):
    """Create LLM with config based on provider."""
    from model import _create_chat_model, ApiProvider, get_model_name
    
    # We can mostly ignore api_config details since model.py manages the singleton key
    # But we can respect overriding params if they were passed
    
    return _create_chat_model(
        temperature=api_config.get('temperature', 0.1),
        max_tokens=api_config.get('max_tokens', 4000)
    )


def try_api_keys_with_fallback(all_api_keys, operation_func, context: str, success_message: str = None, error_message: str = None):
    """
    通用API key循环重试函数

    Args:
        all_api_keys: API key配置列表
        operation_func: 要执行的操作函数，参数为(api_config, provider_name, key_index)
        context: 上下文信息，用于日志
        success_message: 成功时的日志消息模板
        error_message: 错误时的日志消息模板

    Returns:
        (result, success) 元组，result是操作结果，success表示是否成功
    """
    from model import APIErrorException

    for key_index, api_config in enumerate(all_api_keys):
        provider_name = "SiliconFlow" if api_config.get('provider') == 'siliconflow' else "Unknown"
        try:
            result = operation_func(api_config, provider_name, key_index)

            # 成功处理
            if success_message:
                log_with_timestamp(success_message.format(
                    context=context,
                    provider=provider_name,
                    key_num=api_config.get('key_index', key_index) + 1,
                ))
            return result, True
        except APIErrorException as e:
            # API错误，继续下一个key
            if error_message:
                log_with_timestamp(error_message.format(
                    context=context,
                    provider=provider_name,
                    key_num=api_config.get('key_index', key_index) + 1,
                    error=str(e)
                ))
            continue
        except Exception as e:
            # 其他错误，继续下一个key
            log_with_timestamp(f"❌ Unexpected error with {provider_name} Key #{api_config.get('key_index', key_index) + 1}: {e}")
            continue

    # 所有key都失败了
    return None, False