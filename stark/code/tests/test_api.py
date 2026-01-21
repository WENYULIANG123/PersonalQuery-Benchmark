#!/usr/bin/env python3
"""
简单的API测试脚本 - 测试SiliconFlow LLM API连接
"""

import sys
import time
import os

# 添加model.py的路径
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)

try:
    from model import get_gm_model
    print("✅ 成功导入 get_gm_model")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

def test_api_connection():
    """测试API连接"""
    print("\n🔍 测试API连接...")

    try:
        start_time = time.time()
        llm_model = get_gm_model()
        init_time = time.time() - start_time
        print(".2f")
        print(f"✅ LLM模型初始化成功: {type(llm_model).__name__}")

        # 测试简单的API调用
        print("\n🤖 测试API调用...")
        test_prompt = "Hello, please respond with 'API test successful' if you can read this."

        call_start = time.time()
        response = llm_model.invoke(test_prompt)
        call_time = time.time() - call_start

        print(".2f")
        print(f"📝 响应长度: {len(response.content)} 字符")

        # 检查响应内容
        if "successful" in response.content.lower() or "api test" in response.content.lower():
            print("✅ API响应内容正确")
        else:
            print("⚠️ API响应内容可能有问题")
            print(f"📄 响应内容: {response.content[:200]}...")

        return True

    except Exception as e:
        print(f"❌ API测试失败: {e}")
        print(f"❌ 错误类型: {type(e).__name__}")
        return False

def test_network_connectivity():
    """测试网络连接"""
    print("\n🌐 测试网络连接...")
    try:
        import requests
        response = requests.get("https://api.siliconflow.com/v1/models", timeout=10)
        if response.status_code == 200:
            print("✅ SiliconFlow API端点可访问")
            return True
        else:
            print(f"⚠️ API端点响应状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 网络连接测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 SiliconFlow API 测试脚本")
    print("=" * 50)

    # 检查环境变量
    use_siliconflow = os.getenv('USE_SILICONFLOW', 'true').lower() == 'true'
    print(f"📊 USE_SILICONFLOW: {use_siliconflow}")

    if not use_siliconflow:
        print("📝 使用MOCK模式 - 跳过网络测试")
        # 测试mock模式下的模型初始化
        try:
            llm_model = get_gm_model()
            print("✅ Mock模式下模型初始化成功")
            return True
        except Exception as e:
            print(f"❌ Mock模式测试失败: {e}")
            return False

    # 测试网络连接
    network_ok = test_network_connectivity()

    if not network_ok:
        print("\n❌ 网络连接问题，请检查网络设置")
        print("💡 建议：设置环境变量 USE_SILICONFLOW=false 来使用mock模式")
        return False

    # 测试API连接
    api_ok = test_api_connection()

    if api_ok:
        print("\n🎉 所有测试通过！API可以正常使用")
        return True
    else:
        print("\n❌ API测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
