#!/usr/bin/env python3
"""快速验证 fail-fast 错误处理 - 处理 2 个产品"""

import sys
import json
import os
from datetime import datetime

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")
from llm_client import LLMClient

spec = {
    "USER_ID": "A13OFOB1394G31",
    "INPUT_DIR": "/fs04/ar57/wenyu/result/personal_query/00_data_preparation",
    "OUTPUT_DIR": "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction",
}

def safe_str_len(value, context=""):
    import math
    if value is None:
        raise TypeError(f"[{context}] Expected string, got None")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"[{context}] String contains NaN/Inf: {value}")
        raise TypeError(f"[{context}] Expected string, got float: {value}")
    if isinstance(value, str):
        return len(value)
    raise TypeError(f"[{context}] Expected string, got {type(value).__name__}: {str(value)[:50]}")

input_file = os.path.join(spec["INPUT_DIR"], f"reviews_{spec['USER_ID']}.json")
print(f"[{datetime.now().isoformat()}] 📂 加载输入: {input_file}")

with open(input_file) as f:
    data = json.load(f)

results = data.get('results', [])[:3]
print(f"[{datetime.now().isoformat()}] 🧪 处理 {len(results)} 个产品 (共 {data.get('total_products', '?')} 个)")

error_count = 0
success_count = 0

for i, product in enumerate(results):
    asin = product.get('asin', 'UNKNOWN')
    text = product.get('target_review', None)
    
    try:
        length = safe_str_len(text, f"product_{asin}")
        print(f"[{datetime.now().isoformat()}] ✅ [{i+1}] {asin}: reviewText长度={length}")
        success_count += 1
    except (TypeError, ValueError) as e:
        print(f"[{datetime.now().isoformat()}] ❌ [{i+1}] {asin}: {type(e).__name__}: {e}")
        error_count += 1

print(f"[{datetime.now().isoformat()}] 📊 结果: {success_count} 成功, {error_count} 错误")
print(f"[{datetime.now().isoformat()}] ✨ fail-fast验证完成")
