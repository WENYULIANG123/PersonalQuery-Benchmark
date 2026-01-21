#!/usr/bin/env python3
"""
解析 api_raw_responses.json 文件，生成与 user_preference_entities.json 相同格式的内容
"""

import json
import re
from typing import Dict, Optional, Tuple
from collections import defaultdict

def normalize_category_label(category: str) -> str:
    """
    Normalize category labels to keep keys consistent across pipeline.
    Currently enforces: "Color/Finish" -> "Color" (and common variants).
    """
    if category is None:
        return category
    c = str(category).strip()
    if not c:
        return c
    c_lower = c.lower().strip()
    c_compact = c_lower.replace(" ", "")
    if c_compact in {"color/finish", "colour/finish", "colorfinish", "colourfinish"}:
        return "Color"
    if c_lower in {"color", "colour"}:
        return "Color"
    return c

def extract_json_from_content(content: str) -> Optional[Dict]:
    """从响应内容中提取JSON对象"""
    if not content:
        return None
    
    try:
        # 清理内容
        content = content.strip()
        
        # 查找JSON代码块
        json_blocks = []
        start = 0
        while True:
            json_start = content.find('```json', start)
            if json_start == -1:
                break
            json_end = content.find('```', json_start + 7)
            if json_end == -1:
                break
            content_start = content.find('\n', json_start) + 1
            if content_start > 0:
                json_content = content[content_start:json_end].strip()
                if json_content:
                    json_blocks.append(json_content)
            start = json_end + 3
        
        # 如果没有找到json代码块，尝试查找普通代码块
        if not json_blocks:
            if '```' in content:
                last_triple = content.rfind('```')
                first_triple = content.rfind('```', 0, last_triple)
                if first_triple != last_triple:
                    content_start = content.find('\n', first_triple) + 1
                    if content_start > 0:
                        json_content = content[content_start:last_triple].strip()
                        if json_content:
                            json_blocks.append(json_content)
        
        # 如果还是没有找到，尝试直接解析最后几行
        if not json_blocks:
            lines = content.strip().split('\n')
            for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1):
                line = lines[i].strip()
                if line.startswith('{') and line.endswith('}'):
                    json_blocks.append(line)
                    break
        
        # 解析最后一个JSON块
        if json_blocks:
            json_str = json_blocks[-1]
            return json.loads(json_str)
        
        # 如果都没有，尝试直接解析整个内容
        return json.loads(content)
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析错误: {e}")
        return None
    except Exception as e:
        print(f"⚠️ 提取JSON时出错: {e}")
        return None

def extract_review_from_prompt(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """从prompt中提取标题和评论文本"""
    if not prompt:
        return None, None
    
    # 查找 "Text: Title: ... Review: ..." 格式
    text_match = re.search(r'Text:\s*Title:\s*(.+?)\s*Review:\s*(.+?)(?:\n|$)', prompt, re.DOTALL)
    if text_match:
        title = text_match.group(1).strip()
        review = text_match.group(2).strip()
        return title, review
    
    # 如果没有找到，尝试其他格式
    title_match = re.search(r'Title:\s*(.+?)(?:\n|Review:)', prompt, re.DOTALL)
    review_match = re.search(r'Review:\s*(.+?)(?:\n|$)', prompt, re.DOTALL)
    
    title = title_match.group(1).strip() if title_match else None
    review = review_match.group(1).strip() if review_match else None
    
    return title, review

def match_review_to_product(title: str, review_text: str, user_preference_data: Dict) -> Optional[str]:
    """通过匹配评论文本找到对应的ASIN"""
    if not title and not review_text:
        return None
    
    # 创建搜索关键词
    title_lower = title.lower().strip() if title else ""
    review_lower = review_text.lower().strip() if review_text else ""
    
    # 提取关键词（前50个字符的标题和前100个字符的评论）
    title_key = title_lower[:50] if title_lower else ""
    review_key = review_lower[:100] if review_lower else ""
    
    # 在user_preference_data中查找匹配的评论
    best_match = None
    best_score = 0
    
    for product in user_preference_data.get('products', []):
        for review in product.get('review_content', []):
            review_title = review.get('summary', '').strip().lower()
            review_text_content = review.get('reviewText', '').strip().lower()
            
            score = 0
            
            # 匹配标题
            if title_key and review_title:
                # 检查标题是否匹配（至少匹配前30个字符）
                if title_key[:30] in review_title or review_title[:30] in title_key:
                    score += 2
            
            # 匹配评论文本
            if review_key and review_text_content:
                # 检查评论是否匹配（至少匹配前80个字符）
                if review_key[:80] in review_text_content or review_text_content[:80] in review_key:
                    score += 3
            
            # 如果标题和评论都匹配，分数更高
            if score > best_score:
                best_score = score
                best_match = product.get('asin')
    
    # 如果分数足够高，返回匹配的ASIN
    if best_score >= 2:
        return best_match
    
    return None

def parse_api_responses(api_responses_file: str, user_preference_file: str, output_file: str):
    """解析API响应并生成user_preference_entities格式的数据"""
    
    # 读取API响应文件
    print(f"📖 读取API响应文件: {api_responses_file}")
    try:
        with open(api_responses_file, 'r', encoding='utf-8') as f:
            all_responses = json.load(f)
    except Exception as e:
        print(f"❌ 读取API响应文件失败: {e}")
        return
    
    # 读取现有的user_preference_entities文件以获取用户ID和产品信息
    print(f"📖 读取用户偏好文件: {user_preference_file}")
    try:
        with open(user_preference_file, 'r', encoding='utf-8') as f:
            user_preference_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取用户偏好文件失败: {e}")
        return
    
    user_id = user_preference_data.get('user_id', '')
    
    # 过滤成功的响应
    filtered_responses = [
        r for r in all_responses 
        if r.get('context') == 'user_preference_extraction' 
        and r.get('success', False)
    ]
    
    print(f"✅ 找到 {len(filtered_responses)} 个成功的响应")
    
    # 按ASIN组织响应
    asin_responses = defaultdict(list)
    
    for idx, response_data in enumerate(filtered_responses):
        try:
            # 提取评论文本
            prompt = response_data.get('prompt', '')
            title, review_text = extract_review_from_prompt(prompt)
            
            # 匹配到ASIN
            asin = match_review_to_product(title, review_text, user_preference_data)
            
            if not asin:
                # 如果无法匹配，尝试从prompt中查找ASIN
                asin_match = re.search(r'asin["\']?\s*:\s*["\']?([A-Z0-9]{10})', prompt, re.IGNORECASE)
                if asin_match:
                    asin = asin_match.group(1).upper()
            
            if not asin:
                print(f"⚠️ 无法找到响应 {idx} 对应的ASIN (title: {title[:30] if title else 'None'}...)")
                continue
            
            # 解析响应内容
            raw_response = response_data.get('raw_response', {})
            content = raw_response.get('content', '')
            
            if not content:
                print(f"⚠️ 响应 {idx} 的内容为空")
                continue
            
            # 提取JSON实体
            entities = extract_json_from_content(content)
            
            if entities:
                asin_responses[asin].append({
                    'entities': entities,
                    'title': title,
                    'review_text': review_text,
                    'response_data': response_data
                })
            else:
                print(f"⚠️ 无法解析响应 {idx} 的JSON内容")
                
        except Exception as e:
            print(f"⚠️ 处理响应 {idx} 时出错: {e}")
            continue
    
    print(f"✅ 成功解析 {len(asin_responses)} 个产品的响应")
    
    # 构建输出数据
    output_data = {
        'user_id': user_id,
        'products': []
    }
    
    # 为每个ASIN合并实体
    for asin, responses in asin_responses.items():
        # 合并所有响应中的实体
        merged_entities = {}
        
        for response_info in responses:
            entities = response_info['entities']
            if isinstance(entities, dict):
                for category, entity_list in entities.items():
                    category = normalize_category_label(category)
                    if isinstance(entity_list, list):
                        if category not in merged_entities:
                            merged_entities[category] = []
                        # 添加新实体（去重）
                        for entity in entity_list:
                            entity_text = None
                            if isinstance(entity, str):
                                entity_text = entity.strip()
                            elif isinstance(entity, dict):
                                entity_text = str(entity.get("entity") or entity.get("text") or entity.get("name") or "").strip()

                            if entity_text and entity_text not in merged_entities[category]:
                                merged_entities[category].append(entity_text)
        
        # 查找对应的产品评论
        product_reviews = []
        for product in user_preference_data.get('products', []):
            if product.get('asin') == asin:
                product_reviews = product.get('review_content', [])
                break
        
        # 添加到输出
        output_data['products'].append({
            'asin': asin,
            'user_preference_entities': merged_entities,
            'review_content': product_reviews
        })
    
    # 保存输出文件
    print(f"💾 保存结果到: {output_file}")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 成功保存 {len(output_data['products'])} 个产品的数据")
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")

if __name__ == '__main__':
    import sys
    
    api_responses_file = '/home/wlia0047/ar57/wenyu/result/api_raw_responses.json'
    user_preference_file = '/home/wlia0047/ar57/wenyu/result/user_preference_entities.json'
    output_file = '/home/wlia0047/ar57/wenyu/result/user_preference_entities_parsed.json'
    
    if len(sys.argv) > 1:
        api_responses_file = sys.argv[1]
    if len(sys.argv) > 2:
        user_preference_file = sys.argv[2]
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    parse_api_responses(api_responses_file, user_preference_file, output_file)
