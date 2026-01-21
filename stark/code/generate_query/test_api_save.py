#!/usr/bin/env python3
"""
测试脚本：验证API原始响应保存功能
"""

import os
import sys
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_gm_model, call_llm_with_retry, set_api_responses_file

def test_api_response_saving():
    """测试API响应保存功能"""
    print("=" * 60)
    print("🧪 测试API原始响应保存功能")
    print("=" * 60)
    
    # 设置保存路径
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    result_dir = os.path.join(workspace_root, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    api_responses_file = os.path.join(result_dir, "api_raw_responses_test.json")
    set_api_responses_file(api_responses_file)
    print(f"📁 API响应将保存到: {api_responses_file}")
    print()
    
    # 初始化模型
    print("🤖 初始化LLM模型...")
    try:
        llm_model = get_gm_model()
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return False
    
    print()
    
    # 测试1: 简单查询
    print("📝 测试1: 简单查询")
    print("-" * 60)
    test_prompt_1 = "请用一句话介绍Python编程语言。"
    print(f"Prompt: {test_prompt_1}")
    
    try:
        response, success = call_llm_with_retry(
            llm_model, 
            test_prompt_1, 
            max_retries=1, 
            context="test_simple_query",
            use_openai_client=True
        )
        if success:
            print("✅ 调用成功")
            print(f"响应内容: {response}")
            print(f"响应长度: {len(response)} 字符")
        else:
            print("❌ 调用失败")
    except Exception as e:
        print(f"❌ 调用异常: {e}")
    
    print()
    
    # 测试2: JSON格式查询
    print("📝 测试2: JSON格式查询")
    print("-" * 60)
    test_prompt_2 = """请返回一个JSON对象，包含以下信息：
{
  "name": "测试产品",
  "price": 100,
  "category": "电子产品"
}"""
    print(f"Prompt: {test_prompt_2[:50]}...")
    
    try:
        response, success = call_llm_with_retry(
            llm_model, 
            test_prompt_2, 
            max_retries=1, 
            context="test_json_query",
            use_openai_client=True
        )
        if success:
            print("✅ 调用成功")
            print(f"响应内容: {response}")
            print(f"响应长度: {len(response)} 字符")
        else:
            print("❌ 调用失败")
    except Exception as e:
        print(f"❌ 调用异常: {e}")
    
    print()
    
    # 验证保存的文件
    print("📋 验证保存的文件")
    print("-" * 60)
    
    if os.path.exists(api_responses_file):
        print(f"✅ 文件存在: {api_responses_file}")
        
        try:
            with open(api_responses_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            print(f"📊 保存了 {len(saved_data)} 条API调用记录")
            print()
            
            for idx, record in enumerate(saved_data, 1):
                print(f"记录 {idx}:")
                print(f"  时间戳: {record.get('timestamp', 'N/A')}")
                print(f"  场景: {record.get('context', 'N/A')}")
                print(f"  API信息: {record.get('api_info', 'N/A')}")
                print(f"  成功: {record.get('success', False)}")
                print(f"  Prompt长度: {record.get('prompt_length', 0)}")
                print(f"  响应长度: {record.get('response_length', 0)}")
                print(f"  推理长度: {record.get('reasoning_length', 0)}")
                
                # 显示详细的响应内容
                raw_response = record.get('raw_response', {})
                if isinstance(raw_response, dict):
                    reasoning_content = raw_response.get('reasoning_content', '')
                    content = raw_response.get('content', '')
                    
                    if reasoning_content:
                        print(f"  推理内容: {reasoning_content[:200]}{'...' if len(reasoning_content) > 200 else ''}")
                    else:
                        print(f"  推理内容: (空)")
                    
                    if content:
                        print(f"  回复内容: {content[:200]}{'...' if len(content) > 200 else ''}")
                    else:
                        print(f"  回复内容: (空)")
                else:
                    # 兼容旧格式
                    print(f"  原始响应: {str(raw_response)[:200]}...")
                
                if record.get('error'):
                    print(f"  错误: {record.get('error', 'N/A')}")
                print()
            
            print("✅ 所有API调用记录已成功保存！")
            return True
            
        except Exception as e:
            print(f"❌ 读取保存文件失败: {e}")
            return False
    else:
        print(f"❌ 文件不存在: {api_responses_file}")
        return False

if __name__ == "__main__":
    success = test_api_response_saving()
    print()
    print("=" * 60)
    if success:
        print("🎉 测试完成！")
    else:
        print("⚠️ 测试过程中出现问题")
    print("=" * 60)
    sys.exit(0 if success else 1)
