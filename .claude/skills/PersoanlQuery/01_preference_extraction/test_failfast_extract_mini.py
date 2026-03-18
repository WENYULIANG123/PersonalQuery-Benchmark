#!/usr/bin/env python3
"""最小化fail-fast提取测试 - 处理前10个产品"""

import sys
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")
from llm_client import LLMClient
import math

def safe_str_len(value, context=""):
    if value is None:
        raise TypeError(f"[{context}] Expected string, got None")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"[{context}] String contains NaN/Inf")
        raise TypeError(f"[{context}] Expected string, got float")
    if isinstance(value, str):
        return len(value)
    raise TypeError(f"[{context}] Expected string, got {type(value).__name__}")

def process_product(product, user_id=""):
    asin = product.get('asin', 'UNKNOWN')
    
    try:
        text = product.get('target_review')
        text_len = safe_str_len(text, f"product_{asin}")
        return {'asin': asin, 'status': 'OK', 'text_len': text_len}
    except (TypeError, ValueError) as e:
        return {'asin': asin, 'status': 'ERROR', 'error': f"{type(e).__name__}: {str(e)[:100]}"}
    except Exception as e:
        return {'asin': asin, 'status': 'UNKNOWN_ERROR', 'error': str(e)[:100]}

USER_ID = "A13OFOB1394G31"
input_file = f"/fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{USER_ID}.json"

print(f"[{datetime.now().isoformat()}] 🧪 开始迷你fail-fast测试")
print(f"[{datetime.now().isoformat()}] 加载输入文件...")

with open(input_file) as f:
    data = json.load(f)

results = data.get('results', [])[:10]
print(f"[{datetime.now().isoformat()}] 📊 处理 {len(results)} 个产品")

ok_count = 0
error_count = 0

for i, product in enumerate(results, 1):
    result = process_product(product, USER_ID)
    if result['status'] == 'OK':
        ok_count += 1
        print(f"[{datetime.now().isoformat()}] ✅ [{i}] {result['asin']}: text_len={result['text_len']}")
    else:
        error_count += 1
        print(f"[{datetime.now().isoformat()}] ❌ [{i}] {result['asin']}: {result['error']}")

print(f"[{datetime.now().isoformat()}] 📊 完成: {ok_count}✅ {error_count}❌")
print(f"[{datetime.now().isoformat()}] ✨ fail-fast逻辑验证完成")
