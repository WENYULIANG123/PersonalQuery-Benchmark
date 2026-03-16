#!/usr/bin/env python3
"""
Stage 12: Human Evaluation - Generate Human Evaluation Tasks

This script generates human evaluation tasks aligned with Stage 11 LLM evaluation.
It creates an HTML interface where humans can evaluate query-persona alignment
using the same dimension-specific criteria that LLM used.

Input:
- Stage 11 evaluation results (LLM scores)
- Stage 4 personas (user persona descriptions)
- Stage 7 dual queries (original queries)

Output:
- human_eval_tasks.json - Task definitions with dimension-specific info
- evaluation_interface.html - Interactive HTML interface for human evaluation
"""

import json
import os
import sys
import argparse
import random
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")


def log_with_timestamp(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# =============================================================================
# Dimension-Specific Evaluation Rules (aligned with Stage 11)
# =============================================================================

DIMENSION_EVALUATION_RULES = {
    # Category 1: Product Attributes
    "Product_Category": {
        "description": "Product type/name",
        "rule": "Does the query specify or imply a product type that aligns with the user's product preference?",
    },
    "Functionality": {
        "description": "Product features/capabilities",
        "rule": "Does the query mention product features or capabilities that the user wants?",
    },
    "Material_Composition": {
        "description": "Raw material/ingredient",
        "rule": "Does the query specify material preferences that match the user's preference?",
    },
    
    # Category 2: Quality Attributes
    "Quality_Craftsmanship": {
        "description": "Quality/workmanship",
        "rule": "Does the query express quality requirements that align with the user's quality expectations?",
    },
    "Performance": {
        "description": "Performance effectiveness",
        "rule": "Does the query mention performance expectations that match the user's needs?",
    },
    "Safety": {
        "description": "Safety requirements",
        "rule": "Does the query address safety concerns that match the user's safety preferences?",
    },
    
    # Category 3: Appearance/Design
    "Appearance_Color": {
        "description": "Visual appearance",
        "rule": "Does the query specify color or appearance preferences that match the user's taste?",
    },
    "Size_Dimensions": {
        "description": "Size fit",
        "rule": "Does the query specify size requirements that match the user's dimensional needs?",
    },
    "Style_Design": {
        "description": "Style preference",
        "rule": "Does the query express style preferences that match the user's design taste?",
    },
    
    # Category 4: User Experience
    "Comfort": {
        "description": "Comfort level",
        "rule": "Does the query mention comfort requirements that match the user's comfort preferences?",
    },
    "Ease_of_Use": {
        "description": "Usability",
        "rule": "Does the query express ease-of-use requirements that match the user's preference for simplicity?",
    },
    "Portability": {
        "description": "Portability",
        "rule": "Does the query mention portability needs that match the user's mobility requirements?",
    },
    
    # Category 5: Usage Scenarios
    "Target_User": {
        "description": "Intended user",
        "rule": "Does the query indicate it is for a specific user type that matches who the persona is?",
    },
    "Usage_Scenario": {
        "description": "Where/how to use",
        "rule": "Does the query mention usage context that matches the user's actual use case?",
    },
    "Special_Purpose": {
        "description": "Special use case",
        "rule": "Does the query mention a special purpose that aligns with the user's specific needs?",
    },
    
    # Category 6: Price/Value
    "Price": {
        "description": "Price related",
        "rule": "Does the query express price expectations that match the user's budget?",
    },
    "Value": {
        "description": "Value for money",
        "rule": "Does the query mention value considerations that match the user's value orientation?",
    },
    "Packaging_Quantity": {
        "description": "Package size/quantity",
        "rule": "Does the query specify packaging or quantity preferences that match the user's buying habits?",
    },
    
    # Category 7: Special Requirements
    "Compatibility": {
        "description": "Compatibility with existing",
        "rule": "Does the query mention compatibility requirements that match the user's existing setup?",
    },
    "Special_User_Needs": {
        "description": "Special needs",
        "rule": "Does the query address special needs that match the user's unique requirements?",
    },
    "Brand_Preference": {
        "description": "Brand preference",
        "rule": "Does the query mention brand preferences that match the user's brand loyalty?",
    },
}


def load_stage11_evaluations(stage11_dir):
    """Load all Stage 11 LLM evaluation results"""
    log_with_timestamp("Loading Stage 11 LLM evaluations...")
    evaluations = {}
    
    if not os.path.exists(stage11_dir):
        log_with_timestamp(f"Warning: Stage 11 directory not found: {stage11_dir}")
        return evaluations
    
    for filename in os.listdir(stage11_dir):
        if filename.startswith('evaluation_') and filename.endswith('.json') and filename != 'evaluation_summary.json':
            filepath = os.path.join(stage11_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    user_id = data.get('user_id')
                    if user_id:
                        evaluations[user_id] = data
            except Exception as e:
                log_with_timestamp(f"Error loading {filename}: {e}")
    
    log_with_timestamp(f"Loaded {len(evaluations)} user evaluations from Stage 11")
    return evaluations


def load_stage4_personas(persona_dir):
    """Load Stage 4 user personas - supports both per-category and aggregated formats"""
    log_with_timestamp("Loading Stage 4 user personas...")
    personas = {}
    
    if not os.path.exists(persona_dir):
        log_with_timestamp(f"Warning: Persona directory not found: {persona_dir}")
        return personas
    
    for filename in os.listdir(persona_dir):
        if filename.startswith('persona_') and filename.endswith('.json'):
            filepath = os.path.join(persona_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    user_id = data.get('user_id')
                    category = data.get('category', 'Unknown')
                    
                    if user_id:
                        # Store as {user_id: {category: data}}
                        if user_id not in personas:
                            personas[user_id] = {}
                        personas[user_id][category] = data
            except Exception as e:
                log_with_timestamp(f"Error loading {filename}: {e}")
    
    log_with_timestamp(f"Loaded {len(personas)} user personas from Stage 4")
    return personas


def get_user_persona_summary(personas_dict):
    """Generate a summary persona from all category personas"""
    if not personas_dict:
        return "No persona available"
    
    # Combine all dimension personas into a summary
    all_dimensions = {}
    for category, data in personas_dict.items():
        dimension_personas = data.get('dimension_personas', {})
        for dim, persona in dimension_personas.items():
            if dim not in all_dimensions:
                all_dimensions[dim] = persona
    
    # Return a combined summary
    if not all_dimensions:
        return "No persona available"
    
    # Take first 3 dimensions as summary (to avoid too long text)
    sample_dims = list(all_dimensions.keys())[:3]
    summary_parts = []
    for dim in sample_dims:
        summary_parts.append(f"[{dim}]: {all_dimensions[dim][:200]}...")
    
    return "\n\n".join(summary_parts)


def load_stage7_dual_queries(dual_queries_dir):
    """Load Stage 7 dual queries"""
    log_with_timestamp("Loading Stage 7 dual queries...")
    dual_queries = {}
    
    if not os.path.exists(dual_queries_dir):
        log_with_timestamp(f"Warning: Dual queries directory not found: {dual_queries_dir}")
        return dual_queries
    
    for filename in os.listdir(dual_queries_dir):
        if filename.startswith('dual_queries_') and filename.endswith('.json'):
            filepath = os.path.join(dual_queries_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    user_id = data.get('user_id')
                    if user_id:
                        dual_queries[user_id] = data
            except Exception as e:
                log_with_timestamp(f"Error loading {filename}: {e}")
    
    log_with_timestamp(f"Loaded {len(dual_queries)} dual query files from Stage 7")
    return dual_queries


def get_dimension_rule_html(dimension):
    """Get the evaluation rule for a dimension in HTML format"""
    if dimension in DIMENSION_EVALUATION_RULES:
        return DIMENSION_EVALUATION_RULES[dimension]["rule"]
    return f"Does the query align with the user's {dimension} preference?"


def generate_evaluation_tasks(evaluations, personas, dual_queries, sample_size=None):
    """Generate human evaluation tasks aligned with Stage 11"""
    log_with_timestamp("Generating human evaluation tasks...")
    tasks = []
    stats = {
        'total_users': 0,
        'total_query_pairs': 0,
        'users_with_missing_data': []
    }
    
    for user_id in evaluations.keys():
        eval_data = evaluations[user_id]
        
        # Get evaluation results
        eval_results = eval_data.get('results', [])
        
        # Optionally sample results
        if sample_size and len(eval_results) > sample_size:
            eval_results = random.sample(eval_results, sample_size)
        
        for eval_result in eval_results:
            asin = eval_result.get('asin', 'Unknown')
            category = eval_result.get('category', 'Unknown')
            dimensions = eval_result.get('shared_dimensions', [])
            dimension_personas = eval_result.get('dimension_personas', {})
            
            # Get query texts
            target_query = eval_result.get('target_user_query', '')
            mass_query = eval_result.get('mass_market_query', '')
            
            if not target_query or not mass_query:
                continue
            
            # Get LLM scores
            target_eval = eval_result.get('target_evaluation', {})
            mass_eval = eval_result.get('mass_market_evaluation', {})
            
            target_score = target_eval.get('total_score', 0)
            mass_score = mass_eval.get('total_score', 0)
            
            # Determine LLM preference
            if target_score > mass_score:
                llm_prefers = 'personalized'
            elif mass_score > target_score:
                llm_prefers = 'public'
            else:
                llm_prefers = 'tie'
            
            # Build dimension-specific info for human evaluation
            dimension_info = []
            for dim in dimensions:
                dim_rule = get_dimension_rule_html(dim)
                dim_persona = dimension_personas.get(dim, '')
                
                # Get LLM's per-dimension judgment
                dim_key = f"{dim}_alignment"
                target_dim_passed = target_eval.get(dim_key, {}).get('passed', None)
                mass_dim_passed = mass_eval.get(dim_key, {}).get('passed', None)
                
                dimension_info.append({
                    'dimension': dim,
                    'rule': dim_rule,
                    'persona': dim_persona,
                    'llm_target_passed': target_dim_passed,
                    'llm_mass_passed': mass_dim_passed
                })
            
            # Generate evaluation task
            if random.random() < 0.5:
                query_a = target_query
                query_b = mass_query
                query_a_type = 'target'
                query_b_type = 'mass'
            else:
                query_a = mass_query
                query_b = target_query
                query_a_type = 'mass'
                query_b_type = 'target'
            
            task = {
                'evaluation_id': f"eval_{datetime.now().strftime('%Y%m%d')}_{user_id}_{asin}",
                'user_id': user_id,
                'asin': asin,
                'category': category,
                'dimensions': dimensions,
                'dimension_info': dimension_info,
                'queries': {
                    'query_a': query_a,
                    'query_b': query_b,
                    'query_a_type': query_a_type,
                    'query_b_type': query_b_type
                },
                'llm_scores': {
                    'target_score': target_score,
                    'mass_score': mass_score,
                    'score_diff': target_score - mass_score,
                    'llm_prefers': llm_prefers,
                    'total_dimensions': len(dimensions)
                },
                'human_evaluation': {
                    'preferred_query': None,  # 'target_user' | 'mass_market' | 'tie'
                    'target_score': None,     # 0-n (number of dimensions passed)
                    'mass_score': None,       # 0-n (number of dimensions passed)
                    'dimension_evaluations': {},  # {dimension: 'pass'|'fail'}
                    'reasoning': ''
                }
            }
            
            tasks.append(task)
            stats['total_query_pairs'] += 1
        
        stats['total_users'] += 1
    
    log_with_timestamp(f"Generated {stats['total_query_pairs']} evaluation tasks from {stats['total_users']} users")
    
    return tasks, stats


def generate_html_interface(tasks, output_path, language='en'):
    """Generate interactive HTML interface for human evaluation
    
    Aligned with Stage 11's dimension-specific evaluation approach.
    """
    log_with_timestamp(f"Generating HTML evaluation interface ({language})...")
    
    # Language strings
    if language == 'zh':
        strings = {
            'title': '🎯 人类评估界面',
            'subtitle': '用户画像查询对齐度评估',
            'completed': '已完成',
            'remaining': '剩余',
            'save_progress': '💾 保存进度',
            'load_progress': '📂 加载进度',
            'clear_all': '🗑️ 清空所有',
            'download_results': '⬇️ 下载结果',
            'toggle_llm': '👁️ 显示/隐藏 LLM 评分',
            'toggle_query_type': '🔓 显示/隐藏 查询类型',
            'query_a': '🔍 查询 A',
            'query_b': '🔍 查询 B',
            'your_evaluation': '📊 您的评估',
            'preference_question': '哪个查询更好地符合用户画像？',
            'query_a_better': '查询 A 更好',
            'query_b_better': '查询 B 更好',
            'tie': '两者相当',
            'dimension_evaluation': '📐 维度评估',
            'dimension_rule': '评估规则',
            'query_a_passed': '查询A通过',
            'query_b_passed': '查询B通过',
            'both_passed': '两者都通过',
            'neither_passed': '都不通过',
            'reasoning': '评估理由 (可选):',
            'reasoning_placeholder': '请说明您的评估理由...',
            'llm_scores_title': '🤖 LLM 评分 (仅供参考)',
            'llm_query_a': '查询 A',
            'llm_query_b': '查询 B',
            'llm_prefers': 'LLM 倾向',
            'status_completed': '✓ 已完成',
            'status_pending': '○ 待评估',
            'save_success': '✅ 进度保存成功！',
            'clear_warning': '⚠️ 确定要清空所有进度吗？此操作无法撤销。',
            'scoring_guide_title': '📏 评分指南',
            'scoring_description': '对于每个维度，判断查询A和查询B是否满足该维度的评估规则。',
            'pass_label': '通过 ✓',
            'fail_label': '未通过 ✗',
            'instructions': '使用下方的复选框为每个维度打分，然后选择整体偏好。',
        }
    else:
        strings = {
            'title': '🎯 Human Evaluation Interface',
            'subtitle': 'User Profile Query Alignment Assessment',
            'completed': 'Completed',
            'remaining': 'Remaining',
            'save_progress': '💾 Save Progress',
            'load_progress': '📂 Load Progress',
            'clear_all': '🗑️ Clear All',
            'download_results': '⬇️ Download Results',
            'toggle_llm': '👁️ Toggle LLM Scores',
            'toggle_query_type': '🔓 Toggle Query Types',
            'query_a': '🔍 Query A',
            'query_b': '🔍 Query B',
            'your_evaluation': '📊 Your Evaluation',
            'preference_question': 'Which query better matches the user\'s persona?',
            'query_a_better': 'Query A is better',
            'query_b_better': 'Query B is better',
            'tie': 'Tie',
            'dimension_evaluation': '📐 Dimension Evaluation',
            'dimension_rule': 'Evaluation Rule',
            'query_a_passed': 'Query A',
            'query_b_passed': 'Query B',
            'both_passed': 'Both',
            'neither_passed': 'Neither',
            'reasoning': 'Reasoning (optional):',
            'reasoning_placeholder': 'Explain your evaluation...',
            'llm_scores_title': '🤖 LLM Scores (for reference only)',
            'llm_query_a': 'Query A',
            'llm_query_b': 'Query B',
            'llm_prefers': 'LLM Prefers',
            'status_completed': '✓ Completed',
            'status_pending': '○ Pending',
            'save_success': '✅ Progress saved successfully!',
            'clear_warning': '⚠️ Are you sure you want to clear progress?',
            'scoring_guide_title': '📏 Scoring Guide',
            'scoring_description': 'For each dimension, evaluate whether Query A and Query B satisfy that dimension\'s evaluation rule.',
            'pass_label': 'Pass ✓',
            'fail_label': 'Fail ✗',
            'instructions': 'Use the checkboxes below to score each dimension, then select your overall preference.',
        }
    
    # Build dimension rules JavaScript
    dimension_rules_js = json.dumps(DIMENSION_EVALUATION_RULES, ensure_ascii=False)
    
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Evaluation - Query Alignment (Stage 11 Aligned)</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }

        .progress-section {
            background: #f8f9fa;
            padding: 20px 30px;
            border-bottom: 2px solid #e9ecef;
        }

        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 10px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }

        .progress-stats {
            display: flex;
            justify-content: space-between;
            color: #495057;
            font-size: 0.9em;
        }

        .controls {
            padding: 20px 30px;
            background: #f8f9fa;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            border-bottom: 2px solid #e9ecef;
        }

        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }

        .btn-secondary {
            background: #6c757d;
            color: white;
        }

        .btn-secondary:hover {
            background: #5a6268;
        }

        .btn-success {
            background: #28a745;
            color: white;
        }

        .btn-success:hover {
            background: #218838;
        }

        .btn-info {
            background: #17a2b8;
            color: white;
        }

        .btn-info:hover {
            background: #138496;
        }

        .btn-warning {
            background: #ffc107;
            color: #212529;
        }

        .btn-warning:hover {
            background: #e0a800;
        }

        .task-container {
            padding: 30px;
        }

        .task {
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            transition: all 0.3s ease;
        }

        .task.completed {
            background: #d4edda;
            border-color: #28a745;
        }

        .task.current {
            border-color: #667eea;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
        }

        .task-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #dee2e6;
        }

        .task-id {
            font-weight: bold;
            color: #495057;
            font-size: 0.9em;
        }

        .task-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }

        .task-status.pending {
            background: #ffc107;
            color: #212529;
        }

        .task-status.completed {
            background: #28a745;
            color: white;
        }

        .persona-section {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
        }

        .persona-section h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .persona-text {
            color: #495057;
            line-height: 1.8;
            max-height: 150px;
            overflow-y: auto;
        }

        .queries-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }

        .query-box {
            background: white;
            border: 3px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            transition: all 0.3s ease;
        }

        .query-box:hover {
            border-color: #adb5bd;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .query-box.target {
            border-color: #28a745;
        }

        .query-box.public {
            border-color: #17a2b8;
        }

        .query-box h4 {
            margin-bottom: 15px;
            font-size: 1.1em;
        }

        .query-box.target h4 {
            color: #28a745;
        }

        .query-box.public h4 {
            color: #17a2b8;
        }

        .query-text {
            color: #212529;
            line-height: 1.8;
            font-style: italic;
        }

        .query-type-label {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
            margin-bottom: 8px;
        }

        .query-type-label:empty {
            display: none;
        }

        .query-box-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .query-box-header h4 {
            margin: 0;
        }

        .toggle-type-btn {
            background: none;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            cursor: pointer;
            padding: 4px 8px;
            font-size: 0.9em;
            transition: all 0.2s ease;
        }

        .toggle-type-btn:hover {
            background: #e9ecef;
            border-color: #adb5bd;
        }

        .dimension-section {
            background: white;
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }

        .dimension-section h3 {
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.1em;
        }

        .instructions {
            background: #e7f3ff;
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 12px 15px;
            margin-bottom: 15px;
            font-size: 0.9em;
            color: #495057;
        }

        .dimension-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }

        .dimension-item {
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
        }

        .dimension-item h5 {
            color: #667eea;
            margin-bottom: 8px;
            font-size: 0.95em;
        }

        .dimension-rule {
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 10px;
            font-style: italic;
        }

        .dimension-persona {
            font-size: 0.85em;
            color: #495057;
            margin-bottom: 10px;
            padding: 8px;
            background: #fff;
            border-radius: 4px;
            border-left: 3px solid #667eea;
        }

        .dimension-radio-group {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .dimension-radio {
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            font-size: 0.9em;
            padding: 8px 12px;
            border: 2px solid #dee2e6;
            border-radius: 6px;
            transition: all 0.2s ease;
        }

        .dimension-radio:hover {
            border-color: #667eea;
        }

        .dimension-radio input {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }

        .dimension-radio:has(input:checked) {
            border-color: #667eea;
            background: #f0f4ff;
        }

        .dimension-checkboxes {
            display: flex;
            gap: 15px;
        }

        .dimension-checkbox {
            display: flex;
            align-items: center;
            gap: 5px;
            cursor: pointer;
            font-size: 0.9em;
        }

        .dimension-checkbox input {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }

        .dimension-checkbox.pass {
            color: #28a745;
        }

        .dimension-checkbox.fail {
            color: #dc3545;
        }

        .evaluation-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            border: 2px solid #dee2e6;
            margin-top: 20px;
        }

        .evaluation-section h3 {
            color: #495057;
            margin-bottom: 20px;
            font-size: 1.2em;
        }

        .human-scores {
            display: flex;
            gap: 30px;
            background: #e3f2fd;
            border: 2px solid #2196f3;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            font-size: 1.1em;
        }

        .human-score-item {
            font-weight: bold;
            color: #1565c0;
        }

        .human-score-a, .human-score-b {
            color: #d32f2f;
            font-size: 1.3em;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 10px;
            color: #495057;
            font-weight: 600;
        }

        .radio-group {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .radio-option {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            padding: 10px 15px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            transition: all 0.2s ease;
        }

        .radio-option:hover {
            border-color: #667eea;
        }

        .radio-option.selected {
            border-color: #667eea;
            background: #f0f4ff;
        }

        .radio-option input[type="radio"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }

        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-family: inherit;
            font-size: 0.95em;
            resize: vertical;
            min-height: 80px;
        }

        textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        .llm-scores {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .llm-scores h4 {
            color: #856404;
            margin-bottom: 10px;
        }

        .llm-scores p {
            color: #856404;
            margin: 5px 0;
        }

        .llm-summary {
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ffc107;
        }

        .llm-dimensions {
            margin-top: 10px;
        }

        .llm-dim-header {
            font-weight: bold;
            margin-bottom: 8px;
        }

        .llm-dim-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
            padding: 3px 0;
            border-bottom: 1px dashed #e9ecef;
        }

        .llm-dimension-scores {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #ffc107;
        }

        .llm-dim-score {
            display: inline-block;
            margin-right: 10px;
            margin-bottom: 5px;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
        }

        .llm-dim-score.pass {
            background: #d4edda;
            color: #155724;
        }

        .llm-dim-score.fail {
            background: #f8d7da;
            color: #721c24;
        }

        .hidden {
            display: none;
        }

        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }

        @media (max-width: 968px) {
            .queries-section {
                grid-template-columns: 1fr;
            }
            .dimension-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{title}}</h1>
            <p>{{subtitle}}</p>
        </div>

        <div class="progress-section">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill">0%</div>
            </div>
            <div class="progress-stats">
                <span>{{completed}}: <strong id="completedCount">0</strong> / <span id="totalCount">0</span></span>
                <span>{{remaining}}: <strong id="remainingCount">0</strong></span>
            </div>
        </div>

        <div class="controls">
            <button class="btn btn-primary" onclick="saveToFile()">{{save_progress}}</button>
            <button class="btn btn-info" onclick="document.getElementById('fileInput').click()">{{load_progress}}</button>
            <input type="file" id="fileInput" accept=".json" style="display: none;" onchange="loadFromFile(event)">
            <button class="btn btn-secondary" onclick="clearAll()">{{clear_all}}</button>
            <button class="btn btn-success" onclick="exportResults()">{{download_results}}</button>
            <button class="btn btn-secondary" onclick="toggleLLMScores()">{{toggle_llm}}</button>
            <button class="btn btn-warning" onclick="toggleQueryTypes()">{{toggle_query_type}}</button>
        </div>

        <div class="task-container" id="taskContainer">
            <!-- Tasks will be inserted here -->
        </div>
    </div>

    <script>
        // Embedded tasks data
        const tasks = __TASKS_DATA__;
        
        // Dimension evaluation rules (from Stage 11)
        const dimensionRules = __DIMENSION_RULES__;
        
        let results = {};

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadFromLocalStorage();
            renderTasks();
            updateProgress();
        });

        function loadFromLocalStorage() {
            const saved = localStorage.getItem('humanEvalResults_Stage11');
            if (saved) {
                results = JSON.parse(saved);
            }
        }

        function saveToLocalStorage() {
            localStorage.setItem('humanEvalResults_Stage11', JSON.stringify(results));
        }

        function renderTasks() {
            const container = document.getElementById('taskContainer');
            container.innerHTML = '';

            tasks.forEach((task, index) => {
                const isCompleted = results[task.evaluation_id]?.completed;
                const taskDiv = document.createElement('div');
                taskDiv.className = `task ${isCompleted ? 'completed' : ''} ${index === 0 && !isCompleted ? 'current' : ''}`;
                taskDiv.id = `task-${index}`;

                // Get saved dimension evaluations
                const savedDims = results[task.evaluation_id]?.dimension_evaluations || {};

                // Build dimension items HTML
                let dimensionsHtml = '';
                task.dimension_info.forEach((dimInfo, dimIdx) => {
                    const savedEval = savedDims[dimInfo.dimension]?.eval;
                    
                    const checkedA = savedEval === 'a' ? 'checked' : '';
                    const checkedB = savedEval === 'b' ? 'checked' : '';
                    const checkedBoth = savedEval === 'both' ? 'checked' : '';
                    const checkedNone = savedEval === 'none' ? 'checked' : '';
                    
                    dimensionsHtml += `
                        <div class="dimension-item">
                            <h5>${dimInfo.dimension}</h5>
                            <p class="dimension-rule"><strong>{{dimension_rule}}:</strong> ${dimInfo.rule}</p>
                            <p class="dimension-persona"><strong>Persona:</strong> ${dimInfo.persona}</p>
                            <div class="dimension-radio-group">
                                <label class="dimension-radio">
                                    <input type="radio" 
                                           name="dim_${task.evaluation_id}_${dimIdx}"
                                           value="a"
                                           ${checkedA}
                                           onchange="updateDimensionEval('${task.evaluation_id}', '${dimInfo.dimension}', 'a')">
                                    {{query_a_passed}}
                                </label>
                                <label class="dimension-radio">
                                    <input type="radio" 
                                           name="dim_${task.evaluation_id}_${dimIdx}"
                                           value="b"
                                           ${checkedB}
                                           onchange="updateDimensionEval('${task.evaluation_id}', '${dimInfo.dimension}', 'b')">
                                    {{query_b_passed}}
                                </label>
                                <label class="dimension-radio">
                                    <input type="radio" 
                                           name="dim_${task.evaluation_id}_${dimIdx}"
                                           value="both"
                                           ${checkedBoth}
                                           onchange="updateDimensionEval('${task.evaluation_id}', '${dimInfo.dimension}', 'both')">
                                    {{both_passed}}
                                </label>
                                <label class="dimension-radio">
                                    <input type="radio" 
                                           name="dim_${task.evaluation_id}_${dimIdx}"
                                           value="none"
                                           ${checkedNone}
                                           onchange="updateDimensionEval('${task.evaluation_id}', '${dimInfo.dimension}', 'none')">
                                    {{neither_passed}}
                                </label>
                            </div>
                        </div>
                    `;
                });

                taskDiv.innerHTML = `
                    <div class="task-header">
                        <span class="task-id">${task.evaluation_id}</span>
                        <span class="task-status ${isCompleted ? 'completed' : 'pending'}">
                            ${isCompleted ? '{{status_completed}}' : '{{status_pending}}'}
                        </span>
                    </div>

                    <div class="queries-section">
                        <div class="query-box target">
                            <div class="query-box-header">
                                <h4>{{query_a}}</h4>
                                <button class="toggle-type-btn" onclick="toggleQueryType(this, 'query_a_${task.evaluation_id}')">👁️</button>
                            </div>
                            <p class="query-type-label hidden" id="query_a_${task.evaluation_id}">${task.queries.query_a_type === 'target' ? ' Personalized ' : ' Public '}</p>
                            <p class="query-text">"${task.queries.query_a}"</p>
                        </div>
                        <div class="query-box public">
                            <div class="query-box-header">
                                <h4>{{query_b}}</h4>
                                <button class="toggle-type-btn" onclick="toggleQueryType(this, 'query_b_${task.evaluation_id}')">👁️</button>
                            </div>
                            <p class="query-type-label hidden" id="query_b_${task.evaluation_id}">${task.queries.query_b_type === 'target' ? ' Personalized ' : ' Public '}</p>
                            <p class="query-text">"${task.queries.query_b}"</p>
                        </div>
                    </div>

                    <div class="dimension-section">
                        <h3>{{dimension_evaluation}}</h3>
                        <div class="instructions">
                            {{instructions}}
                        </div>
                        <div class="dimension-grid">
                            ${dimensionsHtml}
                        </div>
                    </div>

                    <div class="evaluation-section">
                        <h3>{{your_evaluation}}</h3>

                        <div class="human-scores">
                            <span class="human-score-item"><strong>查询A:</strong> <span class="human-score-a">0</span>/${task.dimensions.length}</span>
                            <span class="human-score-item"><strong>查询B:</strong> <span class="human-score-b">0</span>/${task.dimensions.length}</span>
                        </div>

                        <div class="form-group">
                            <label>{{reasoning}}</label>
                            <textarea id="reasoning-${task.evaluation_id}" placeholder="{{reasoning_placeholder}}"
                                      onblur="updateReasoning('${task.evaluation_id}', this.value)"></textarea>
                        </div>

                        <div class="llm-scores hidden" id="llm-${task.evaluation_id}">
                            <h4>{{llm_scores_title}}</h4>
                            <div class="llm-summary">
                                <p><strong>{{llm_query_a}}:</strong> ${task.llm_scores.target_score}/${task.llm_scores.total_dimensions}</p>
                                <p><strong>{{llm_query_b}}:</strong> ${task.llm_scores.mass_score}/${task.llm_scores.total_dimensions}</p>
                                <p><strong>{{llm_prefers}}:</strong> ${task.llm_scores.llm_prefers}</p>
                                <p><strong>Score Diff:</strong> ${task.llm_scores.score_diff}</p>
                            </div>
                            <div class="llm-dimensions">
                                <p class="llm-dim-header"><strong>Dimension Breakdown:</strong></p>
                                ${task.dimension_info.map(d => {
                                    const targetPass = d.llm_target_passed === true ? '✓' : (d.llm_target_passed === false ? '✗' : '-');
                                    const massPass = d.llm_mass_passed === true ? '✓' : (d.llm_mass_passed === false ? '✗' : '-');
                                    return `<p class="llm-dim-row"><span>${d.dimension}:</span> <span>A=${targetPass} B=${massPass}</span></p>`;
                                }).join('')}
                            </div>
                        </div>
                    </div>
                `;

                container.appendChild(taskDiv);

                // Load saved values
                if (results[task.evaluation_id]) {
                    const result = results[task.evaluation_id];
                    if (result.preferred_query) {
                        const radio = taskDiv.querySelector(`input[name="pref_${task.evaluation_id}"][value="${result.preferred_query}"]`);
                        if (radio) {
                            radio.checked = true;
                            radio.parentElement.classList.add('selected');
                        }
                    }
                    if (result.reasoning) {
                        taskDiv.querySelector('textarea').value = result.reasoning;
                    }
                }
            });

            document.getElementById('totalCount').textContent = tasks.length;
        }

        function updateDimensionEval(taskId, dimension, evalValue) {
            if (!results[taskId]) results[taskId] = {};
            if (!results[taskId].dimension_evaluations) results[taskId].dimension_evaluations = {};
            
            results[taskId].dimension_evaluations[dimension] = { eval: evalValue };
            
            const dimEvals = results[taskId].dimension_evaluations;
            let scoreA = 0;
            let scoreB = 0;
            
            Object.keys(dimEvals).forEach(dim => {
                if (dimEvals[dim].eval === 'a') scoreA++;
                if (dimEvals[dim].eval === 'b') scoreB++;
                if (dimEvals[dim].eval === 'both') {
                    scoreA++;
                    scoreB++;
                }
            });
            
            results[taskId].score_a = scoreA;
            results[taskId].score_b = scoreB;
            
            // Update displayed scores - find task div containing this evaluation
            const allTaskDivs = document.querySelectorAll('.task');
            allTaskDivs.forEach(div => {
                if (div.innerHTML.includes(taskId)) {
                    const scoreAEl = div.querySelector('.human-score-a');
                    const scoreBEl = div.querySelector('.human-score-b');
                    if (scoreAEl) scoreAEl.textContent = scoreA;
                    if (scoreBEl) scoreBEl.textContent = scoreB;
                }
            });
            
            results[taskId].completed = isTaskComplete(taskId);
            saveToLocalStorage();
            updateProgress();
        }

        function updateReasoning(taskId, value) {
            if (!results[taskId]) results[taskId] = {};
            results[taskId].reasoning = value;
            saveToLocalStorage();
        }

        function isTaskComplete(taskId) {
            const result = results[taskId];
            return result && result.dimension_evaluations && 
                   Object.keys(result.dimension_evaluations).length > 0;
        }

        function updateProgress() {
            const completed = Object.values(results).filter(r => r.completed).length;
            const total = tasks.length;
            const percentage = Math.round((completed / total) * 100);

            const progressFill = document.getElementById('progressFill');
            const completedCount = document.getElementById('completedCount');
            const remainingCount = document.getElementById('remainingCount');

            if (progressFill) {
                progressFill.style.width = percentage + '%';
                progressFill.textContent = percentage + '%';
            }
            if (completedCount) {
                completedCount.textContent = completed;
            }
            if (remainingCount) {
                remainingCount.textContent = total - completed;
            }

            // Update task status badges
            tasks.forEach((task, index) => {
                const taskDiv = document.getElementById(`task-${index}`);
                if (taskDiv) {
                    const statusBadge = taskDiv.querySelector('.task-status');
                    if (results[task.evaluation_id]?.completed) {
                        taskDiv.classList.add('completed');
                        if (statusBadge) {
                            statusBadge.textContent = '{{status_completed}}';
                            statusBadge.className = 'task-status completed';
                        }
                    }
                }
            });
        }

        function saveToFile() {
            const exportData = tasks.map(task => ({
                ...task,
                human_evaluation: results[task.evaluation_id] || {
                    preferred_query: null,
                    target_score: null,
                    mass_score: null,
                    dimension_evaluations: {},
                    reasoning: ''
                }
            }));

            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'human_eval_progress.json';
            a.click();
            URL.revokeObjectURL(url);

            const completedCount = Object.keys(results).filter(k => results[k].completed).length;
            alert(`Progress saved!\\n\\nCompleted: ${completedCount}/${tasks.length}`);
        }

        function loadFromFile(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const loadedData = JSON.parse(e.target.result);
                    const loadedResults = {};
                    let loadedCount = 0;

                    loadedData.forEach(item => {
                        if (item.human_evaluation && item.human_evaluation.preferred_query) {
                            loadedResults[item.evaluation_id] = item.human_evaluation;
                            loadedCount++;
                        }
                    });

                    results = loadedResults;
                    saveToLocalStorage();
                    renderTasks();
                    updateProgress();

                    alert(`Progress loaded!\\n\\nRestored: ${loadedCount} evaluations`);
                } catch (error) {
                    alert('Failed to load file: ' + error.message);
                }
            };
            reader.readAsText(file);
            event.target.value = '';
        }

        function clearAll() {
            if (confirm('Are you sure you want to clear all progress?')) {
                localStorage.removeItem('humanEvalResults_Stage11');
                results = {};
                location.reload();
            }
        }

        function exportResults() {
            const exportData = tasks.map(task => ({
                ...task,
                human_evaluation: results[task.evaluation_id] || {
                    preferred_query: null,
                    target_score: null,
                    mass_score: null,
                    dimension_evaluations: {},
                    reasoning: ''
                }
            }));

            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `human_eval_results_${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }

        function toggleLLMScores() {
            document.querySelectorAll('.llm-scores').forEach(el => el.classList.toggle('hidden'));
        }

        function toggleQueryTypes() {
            document.querySelectorAll('.query-type-label').forEach(el => el.classList.toggle('hidden'));
        }

        function toggleQueryType(btn, elementId) {
            const el = document.getElementById(elementId);
            if (el) {
                el.classList.toggle('hidden');
                btn.textContent = el.classList.contains('hidden') ? '👁️' : '🙈';
            }
        }
    </script>
</body>
</html>"""

    # Replace placeholder with actual tasks data
    tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
    html_content = html_template.replace('__TASKS_DATA__', tasks_json)
    
    # Replace dimension rules
    dimension_rules_json = json.dumps(DIMENSION_EVALUATION_RULES, ensure_ascii=False)
    html_content = html_content.replace('__DIMENSION_RULES__', dimension_rules_json)

    # Replace language strings
    for key, value in strings.items():
        html_content = html_content.replace(f'{{{{{key}}}}}', value)

    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    log_with_timestamp(f"HTML interface saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate human evaluation tasks aligned with Stage 11 LLM evaluation"
    )
    parser.add_argument(
        "--stage11-dir",
        default="/fs04/ar57/wenyu/result/personal_query/11_evaluation",
        help="Directory containing Stage 11 LLM evaluation results"
    )
    parser.add_argument(
        "--stage4-dir",
        default="/fs04/ar57/wenyu/result/personal_query/04_persona",
        help="Directory containing Stage 4 persona files"
    )
    parser.add_argument(
        "--stage7-dir",
        default="/fs04/ar57/wenyu/result/personal_query/07_query",
        help="Directory containing Stage 7 dual query files"
    )
    parser.add_argument(
        "--output-dir",
        default="/fs04/ar57/wenyu/result/personal_query/12_human_evaluation/tasks",
        help="Output directory for evaluation tasks and HTML interface"
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=['en', 'zh'],
        help="Interface language: en for English, zh for Chinese"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Maximum number of query pairs per user (for testing)"
    )
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all data sources
    evaluations = load_stage11_evaluations(args.stage11_dir)
    personas = load_stage4_personas(args.stage4_dir)
    dual_queries = load_stage7_dual_queries(args.stage7_dir)

    # Generate evaluation tasks
    tasks, stats = generate_evaluation_tasks(
        evaluations, 
        personas, 
        dual_queries,
        sample_size=args.sample_size
    )

    # Save tasks as JSON
    tasks_json_path = os.path.join(args.output_dir, 'human_eval_tasks.json')
    with open(tasks_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'statistics': stats,
            'tasks': tasks
        }, f, indent=2, ensure_ascii=False)
    log_with_timestamp(f"Tasks saved to: {tasks_json_path}")

    # Generate HTML interface
    html_filename = f'evaluation_interface_{args.language}.html'
    html_path = os.path.join(args.output_dir, html_filename)
    generate_html_interface(tasks, html_path, args.language)

    # Final summary
    log_with_timestamp("\n" + "="*70)
    log_with_timestamp("GENERATION COMPLETE - STAGE 12 ALIGNED WITH STAGE 11")
    log_with_timestamp("="*70)
    log_with_timestamp(f"Total evaluation tasks: {stats['total_query_pairs']}")
    log_with_timestamp(f"Total users: {stats['total_users']}")
    log_with_timestamp(f"\nOutput files:")
    log_with_timestamp(f"  1. {tasks_json_path}")
    log_with_timestamp(f"  2. {html_path}")
    log_with_timestamp(f"\nAlignment with Stage 11:")
    log_with_timestamp(f"  - Uses same dimension-specific evaluation rules")
    log_with_timestamp(f"  - Shows dimension personas to human evaluators")
    log_with_timestamp(f"  - Per-dimension pass/fail scoring (matching LLM)")
    log_with_timestamp(f"  - Total score = number of passed dimensions")
    log_with_timestamp(f"\nNext steps:")
    log_with_timestamp(f"  1. Open {html_path} in your browser")
    log_with_timestamp(f"  2. Complete the evaluation for all {stats['total_query_pairs']} query pairs")
    log_with_timestamp(f"  3. Download the results as JSON")
    log_with_timestamp(f"  4. Run 12_compute_alignment_metrics.py to calculate alignment metrics")
    log_with_timestamp("="*70)


if __name__ == "__main__":
    main()
