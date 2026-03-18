#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, "/home/wlia0047/ar57/wenyu/.claude/skills")
from llm_client import LLMClient

TEST_USER_ID = "ALYZJ7W14YS26"
REVIEW_FILE = f"/fs04/ar57/wenyu/result/personal_query/00_data_preparation/reviews_{TEST_USER_ID}.json"

def log_msg(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

with open(REVIEW_FILE) as f:
    data = json.load(f)

products = data.get('results', [])[:2]
log_msg(f"测试样本：{len(products)} 个产品")

client = LLMClient()

for i, product in enumerate(products):
    asin = product.get('asin', f'product_{i}')
    target_reviews = product.get('target_reviews', [])
    reviews_text = '\n'.join(target_reviews[:1]) if target_reviews else ""
    
    prompt = f"""You are a helpful assistant and an expert in understanding product reviews.
Your task is to extract product aspects and their associated sentiments from customer
reviews, if any are mentioned.

A product aspect refers to a specific feature, attribute, or component of a product or service
that customers mention and evaluate.

The sentiment should be classified as one of: "POSITIVE", "MIXED", or "NEGATIVE".

Output ONLY valid JSON in this format:
{{
  "aspects": [
    {{"aspect": "aspect name", "sentiment": "POSITIVE|MIXED|NEGATIVE"}},
    ...
  ]
}}

Reviews for product {asin}:
{reviews_text[:500]}

Output ONLY the JSON, no other text."""

    log_msg(f"\n{'='*80}")
    log_msg(f"产品 {i+1}: {asin}")
    log_msg(f"评论文本长度: {len(reviews_text)} 字")
    
    try:
        response = client.call(prompt, max_tokens=512)
        log_msg(f"✅ LLM 响应收到")
        log_msg(f"响应长度: {len(response)}")
        log_msg(f"\n📝 原始响应:\n{response}\n")
        
        import re
        try:
            if "```json" in response:
                match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
                    log_msg(f"✅ JSON 解析成功")
                    log_msg(f"提取的方面: {parsed}")
            elif "```" in response:
                match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(1))
                    log_msg(f"✅ JSON 解析成功")
                    log_msg(f"提取的方面: {parsed}")
            else:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    parsed = json.loads(match.group(0))
                    log_msg(f"✅ JSON 解析成功")
                    log_msg(f"提取的方面: {parsed}")
                else:
                    log_msg(f"❌ 无法找到 JSON")
        except json.JSONDecodeError as e:
            log_msg(f"❌ JSON 解析失败: {e}")
    
    except Exception as e:
        log_msg(f"❌ 异常: {e}")

log_msg(f"\n{'='*80}")
log_msg(f"✅ 调试测试完成")
