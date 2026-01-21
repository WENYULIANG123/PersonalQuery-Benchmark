#!/usr/bin/env python3
"""
测试脚本：检查API是否返回reasoning_content
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_gm_model, call_llm_with_retry

def test_reasoning_content():
    """测试reasoning_content返回"""
    print("=" * 60)
    print("🧪 测试reasoning_content返回")
    print("=" * 60)
    
    # 初始化模型
    print("🤖 初始化LLM模型...")
    try:
        llm_model = get_gm_model()
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return False
    
    print()
    
    # 测试查询 - 使用一个需要推理的问题
    print("📝 测试查询（需要推理的问题）")
    print("-" * 60)
    test_prompt = "请解释为什么Python是一种解释型语言，并说明解释型语言和编译型语言的区别。"
    print(f"Prompt: {test_prompt}")
    print()
    
    try:
        print("🔄 调用API（使用OpenAI客户端，thinking_budget=2048）...")
        response, success = call_llm_with_retry(
            llm_model, 
            test_prompt, 
            max_retries=1, 
            context="test_reasoning_content",
            use_openai_client=True
        )
        
        print()
        if success:
            print("✅ 调用成功")
            print(f"响应内容长度: {len(response)} 字符")
            print(f"响应内容: {response[:500]}...")
        else:
            print("❌ 调用失败")
    except Exception as e:
        print(f"❌ 调用异常: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("📋 检查保存的API响应")
    print("=" * 60)
    
    # 检查保存的文件
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    result_dir = os.path.join(workspace_root, "result")
    api_responses_file = os.path.join(result_dir, "api_raw_responses_test.json")
    
    if os.path.exists(api_responses_file):
        try:
            with open(api_responses_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            # 获取最后一条记录
            if saved_data:
                last_record = saved_data[-1]
                print(f"📊 最后一条记录:")
                print(f"  时间戳: {last_record.get('timestamp', 'N/A')}")
                print(f"  场景: {last_record.get('context', 'N/A')}")
                print(f"  成功: {last_record.get('success', False)}")
                
                raw_response = last_record.get('raw_response', {})
                if isinstance(raw_response, dict):
                    reasoning_content = raw_response.get('reasoning_content', '')
                    content = raw_response.get('content', '')
                    
                    print(f"  推理内容长度: {len(reasoning_content)}")
                    if reasoning_content:
                        print(f"  ✅ 有推理内容!")
                        print(f"  推理内容: {reasoning_content[:300]}...")
                    else:
                        print(f"  ⚠️ 推理内容为空")
                    
                    print(f"  回复内容长度: {len(content)}")
                    if content:
                        print(f"  回复内容: {content[:300]}...")
                
                if last_record.get('error'):
                    print(f"  错误: {last_record.get('error', 'N/A')}")
        except Exception as e:
            print(f"❌ 读取保存文件失败: {e}")
    else:
        print(f"⚠️ 文件不存在: {api_responses_file}")

if __name__ == "__main__":
    test_reasoning_content()
