#!/usr/bin/env python3
"""
Stage 11: Human Evaluation - Generate Human Evaluation Tasks

This script generates human evaluation tasks by combining:
- Stage 10 LLM evaluation results
- Stage 9 noisy queries
- Stage 3 user personas

Output:
1. human_eval_tasks.json - Task definitions
2. evaluation_interface.html - Interactive HTML interface for human evaluation
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


def load_stage10_evaluations(stage10_dir):
    """Load all Stage 10 evaluation results"""
    log_with_timestamp("Loading Stage 10 LLM evaluations...")
    evaluations = {}

    for filename in os.listdir(stage10_dir):
        if filename.startswith('evaluation_') and filename.endswith('.json') and filename != 'evaluation_summary.json':
            filepath = os.path.join(stage10_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                user_id = data.get('user_id')
                if user_id:
                    evaluations[user_id] = data

    log_with_timestamp(f"Loaded {len(evaluations)} user evaluations")
    return evaluations


def load_stage9_noisy_queries(stage9_dir):
    """Load all Stage 9 noisy query results"""
    log_with_timestamp("Loading Stage 9 noisy queries...")
    noisy_queries = {}

    for filename in os.listdir(stage9_dir):
        if filename.startswith('noisy_queries_') and filename.endswith('.json') and filename != 'noisy_queries_summary.json':
            filepath = os.path.join(stage9_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                user_id = data.get('user_id')
                if user_id:
                    noisy_queries[user_id] = data

    log_with_timestamp(f"Loaded {len(noisy_queries)} user noisy query files")
    return noisy_queries


def load_stage3_personas(persona_dir):
    """Load all Stage 3 user personas"""
    log_with_timestamp("Loading Stage 3 user personas...")
    personas = {}

    for filename in os.listdir(persona_dir):
        if filename.startswith('persona_') and filename.endswith('.json'):
            filepath = os.path.join(persona_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                user_id = data.get('user_id')
                if user_id:
                    personas[user_id] = data.get('persona', '')

    log_with_timestamp(f"Loaded {len(personas)} user personas")
    return personas


def clean_query_text(text):
    """Remove word count markers and extra quotes from query text"""
    if not text:
        return ''

    # Remove word count markers like "(28 words)"
    import re
    text = re.sub(r'\s*\(\d+\s+words\)\s*\.?\s*$', '', text.strip())

    # Remove leading/trailing quotes if present
    text = text.strip('"\'')

    # Remove extra quotes at the end
    text = re.sub(r'"\s*\(\d+\s+words\)\s*', '', text)

    return text.strip()


def generate_evaluation_tasks(evaluations, noisy_queries, personas):
    """Generate human evaluation tasks by combining all data sources"""
    log_with_timestamp("Generating human evaluation tasks...")
    tasks = []
    stats = {
        'total_users': 0,
        'total_query_pairs': 0,
        'users_with_missing_data': []
    }

    for user_id in evaluations.keys():
        # Check if we have all required data
        if user_id not in noisy_queries:
            stats['users_with_missing_data'].append(f"{user_id} (missing noisy queries)")
            continue
        if user_id not in personas:
            stats['users_with_missing_data'].append(f"{user_id} (missing persona)")
            continue

        eval_data = evaluations[user_id]
        noisy_data = noisy_queries[user_id]
        persona = personas[user_id]

        # Get evaluation results
        eval_results = eval_data.get('results', [])
        noisy_queries_list = noisy_data.get('queries', [])

        # Create a mapping from ASIN to noisy query data
        noisy_map = {q['asin']: q for q in noisy_queries_list}

        # Match evaluation results with noisy queries
        for eval_result in eval_results:
            asin = eval_result.get('asin')
            if asin not in noisy_map:
                continue

            noisy_item = noisy_map[asin]

            # Determine LLM preference
            public_score = eval_result.get('public_score', 0)
            personalized_score = eval_result.get('personalized_score', 0)
            score_diff = eval_result.get('score_diff', 0)

            if personalized_score > public_score:
                llm_prefers = 'personalized'
            elif public_score > personalized_score:
                llm_prefers = 'public'
            else:
                llm_prefers = 'tie'

            # Generate evaluation task
            task = {
                'evaluation_id': f"eval_{datetime.now().strftime('%Y%m%d')}_{user_id}_{asin}",
                'user_id': user_id,
                'user_persona': persona,
                'asin': asin,
                'category': eval_result.get('category', ''),
                'queries': {
                    'public_query_noisy': clean_query_text(eval_result.get('public_query', '')),  # From Stage 10
                    'personalized_query_noisy': clean_query_text(noisy_item.get('personalized_query', {}).get('noisy') or noisy_item.get('personalized_query', {}).get('original', ''))  # From Stage 9
                },
                'llm_scores': {
                    'public_score': public_score,
                    'personalized_score': personalized_score,
                    'score_diff': score_diff,
                    'llm_prefers': llm_prefers
                },
                'human_evaluation': {
                    'preferred_query': None,  # To be filled: 'public' | 'personalized' | 'tie'
                    'public_query_score': None,  # 1-10
                    'personalized_query_score': None,  # 1-10
                    'reasoning': ''
                }
            }

            tasks.append(task)
            stats['total_query_pairs'] += 1

        stats['total_users'] += 1

    log_with_timestamp(f"Generated {stats['total_query_pairs']} evaluation tasks from {stats['total_users']} users")

    if stats['users_with_missing_data']:
        log_with_timestamp(f"Warning: {len(stats['users_with_missing_data'])} users with missing data:")
        for msg in stats['users_with_missing_data'][:5]:  # Show first 5
            log_with_timestamp(f"  - {msg}")

    return tasks, stats


def generate_html_interface(tasks, output_path, language='en'):
    """Generate interactive HTML interface for human evaluation

    Args:
        tasks: List of evaluation tasks
        output_path: Path to save HTML file
        language: 'en' for English, 'zh' for Chinese
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
            'clear_all': '🗑️ 清空所有',
            'download_results': '⬇️ 下载结果',
            'toggle_llm': '👁️ 显示/隐藏 LLM 评分',
            'user_persona': '👤 用户画像',
            'query_a': '🔍 查询 A',
            'query_b': '🔍 查询 B',
            'your_evaluation': '📊 您的评估',
            'preference_question': '哪个查询更符合用户画像？',
            'query_a_better': '查询 A 更好',
            'query_b_better': '查询 B 更好',
            'tie': '两者相当',
            'query_a_score': '查询 A 评分 (1-10):',
            'query_b_score': '查询 B 评分 (1-10):',
            'reasoning': '评估理由 (可选):',
            'reasoning_placeholder': '请说明您的评估理由...',
            'llm_scores_title': '🤖 LLM 评分 (仅供参考)',
            'llm_public': '公共查询',
            'llm_personalized': '个性化查询',
            'llm_prefers': 'LLM 倾向',
            'status_completed': '✓ 已完成',
            'status_pending': '○ 待评估',
            'save_success': '✅ 进度保存成功！',
            'clear_warning': '⚠️ 确定要清空所有进度吗？此操作无法撤销。',
        }
    else:
        strings = {
            'title': '🎯 Human Evaluation Interface',
            'subtitle': 'User Profile Query Alignment Assessment',
            'completed': 'Completed',
            'remaining': 'Remaining',
            'save_progress': '💾 保存进度',
            'load_progress': '📂 加载进度',
            'clear_all': '🗑️ Clear All',
            'download_results': '⬇️ 导出最终结果',
            'toggle_llm': '👁️ Toggle LLM Scores',
            'user_persona': '👤 User Persona',
            'query_a': '🔍 Query A',
            'query_b': '🔍 Query B',
            'your_evaluation': '📊 Your Evaluation',
            'preference_question': 'Which query better matches the user\'s persona?',
            'query_a_better': 'Query A is better',
            'query_b_better': 'Query B is better',
            'tie': 'Tie',
            'query_a_score': 'Query A Score (1-10):',
            'query_b_score': 'Query B Score (1-10):',
            'reasoning': 'Reasoning (optional):',
            'scoring_criteria': '📏 Scoring Criteria (Aligned with LLM Evaluation)',
            'score_1_3': '1-3: Poor - Overly generic query using only broad terms like "high quality", "easy to use", "versatile". Does NOT reflect ANY of the user\'s professional needs or specialized characteristics.',
            'score_4_6': '4-6: Average - Somewhat generic but has some relevance. Lacks specific professional terminology and detailed specifications. OR mentions some professional needs but misses key technical details.',
            'score_7_8': '7-8: Good - Includes professional terminology and reflects the user\'s core professional needs. BUT may lack certain highly specific details (e.g., specific industrial-grade specs, extreme performance requirements).',
            'score_9_10': '9-10: Excellent - Highly specialized query. Uses professional/technical terminology (e.g., "high-tensile strength", "industrial-grade", "abrasion resistance"). Completely AVOIDS generic terms. Precisely aligns with user\'s unique characteristics and work scenarios.',
            'reasoning_placeholder': 'Explain your evaluation...',
            'llm_scores_title': '🤖 LLM Scores (for reference only)',
            'llm_public': 'Public',
            'llm_personalized': 'Personalized',
            'llm_prefers': 'LLM Prefers',
            'status_completed': '✓ Completed',
            'status_pending': '○ Pending',
            'save_success': '✅ Progress saved successfully!',
            'clear_warning': '⚠️ Are you sure you want to clear all progress? This cannot be undone.',
        }

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Evaluation - User Profile Query Alignment</title>
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
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .query-box:hover {
            border-color: #adb5bd;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .query-box.selected {
            border-color: #667eea;
            background: #f0f4ff;
        }

        .query-box h4 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.1em;
        }

        .query-text {
            color: #212529;
            line-height: 1.8;
            font-style: italic;
        }

        .evaluation-section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            border: 2px solid #dee2e6;
        }

        .evaluation-section h3 {
            color: #495057;
            margin-bottom: 20px;
            font-size: 1.2em;
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
        }

        .radio-option input[type="radio"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }

        .score-input {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .score-input input[type="range"] {
            flex: 1;
            height: 8px;
            border-radius: 5px;
            background: #dee2e6;
            outline: none;
            -webkit-appearance: none;
        }

        .score-input input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }

        .score-input input[type="number"] {
            width: 70px;
            padding: 8px;
            border: 2px solid #dee2e6;
            border-radius: 5px;
            text-align: center;
            font-size: 1.1em;
            font-weight: bold;
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
        }

        .llm-scores h4 {
            color: #856404;
            margin-bottom: 10px;
        }

        .llm-scores p {
            color: #856404;
            margin: 5px 0;
        }

        .hidden {
            display: none;
        }

        .scoring-guide {
            background: #f0f7ff;
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 20px;
        }

        .scoring-guide h4 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1em;
        }

        .score-criteria-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .score-criteria-list li {
            padding: 8px 0;
            border-bottom: 1px solid #e0e7ff;
            font-size: 0.9em;
            line-height: 1.4;
        }

        .score-criteria-list li:last-child {
            border-bottom: none;
        }

        .score-criteria-list strong {
            color: #667eea;
            display: inline-block;
            min-width: 50px;
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

        .auto-save-hint {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @media (max-width: 968px) {
            .queries-section {
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
        </div>

        <div class="auto-save-hint" style="text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 10px;">
            💡 所有更改已自动保存到浏览器缓存，刷新页面不会丢失进度。<br>
            💾 使用"保存进度"创建可转移的备份文件，使用"加载进度"恢复数据
        </div>

        <div class="auto-save-hint" style="text-align: center; color: #6c757d; font-size: 0.9em; margin-top: 10px;">
            💡 所有更改已自动保存到浏览器缓存，刷新页面不会丢失进度
        </div>

        <div class="task-container" id="taskContainer">
            <!-- Tasks will be inserted here -->
        </div>
    </div>

    <script>
        // Embedded tasks data
        const tasks = __TASKS_DATA__;
        let results = {};

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadFromLocalStorage();
            renderTasks();
            updateProgress();

            // Show auto-load notification if data was loaded
            const savedCount = Object.keys(results).filter(k => results[k].completed).length;
            if (savedCount > 0) {
                setTimeout(() => {
                    const notification = document.createElement('div');
                    notification.style.cssText = `
                        position: fixed;
                        top: 20px;
                        right: 20px;
                        background: #28a745;
                        color: white;
                        padding: 15px 25px;
                        border-radius: 8px;
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        z-index: 10000;
                        animation: slideIn 0.3s ease-out;
                    `;
                    notification.innerHTML = `✅ 已自动加载 ${savedCount} 个评估进度`;
                    document.body.appendChild(notification);

                    setTimeout(() => {
                        notification.style.opacity = '0';
                        notification.style.transition = 'opacity 0.3s';
                        setTimeout(() => notification.remove(), 300);
                    }, 3000);
                }, 500);
            }
        });

        function loadFromLocalStorage() {
            const saved = localStorage.getItem('humanEvalResults');
            if (saved) {
                results = JSON.parse(saved);
            }
        }

        function saveToLocalStorage() {
            localStorage.setItem('humanEvalResults', JSON.stringify(results));
        }

        function renderTasks() {
            const container = document.getElementById('taskContainer');
            container.innerHTML = '';

            tasks.forEach((task, index) => {
                const isCompleted = results[task.evaluation_id]?.completed;
                const taskDiv = document.createElement('div');
                taskDiv.className = `task ${isCompleted ? 'completed' : ''} ${index === 0 && !isCompleted ? 'current' : ''}`;
                taskDiv.id = `task-${index}`;

                // Randomize query order
                const queries = [
                    { id: 'public', text: task.queries.public_query_noisy },
                    { id: 'personalized', text: task.queries.personalized_query_noisy }
                ];
                const shuffled = [...queries].sort(() => Math.random() - 0.5);

                taskDiv.innerHTML = `
                    <div class="task-header">
                        <span class="task-id">${task.evaluation_id}</span>
                        <span class="task-status ${isCompleted ? 'completed' : 'pending'}">
                            ${isCompleted ? '✓ Completed' : '○ Pending'}
                        </span>
                    </div>

                    <div class="persona-section">
                        <h3>{{user_persona}}</h3>
                        <p class="persona-text">${task.user_persona}</p>
                    </div>

                    <div class="queries-section">
                        <div class="query-box" id="query-a-${index}" onclick="selectQuery('${task.evaluation_id}', 'a')">
                            <h4>{{query_a}}</h4>
                            <p class="query-text">"${shuffled[0].text}"</p>
                        </div>
                        <div class="query-box" id="query-b-${index}" onclick="selectQuery('${task.evaluation_id}', 'b')">
                            <h4>{{query_b}}</h4>
                            <p class="query-text">"${shuffled[1].text}"</p>
                        </div>
                    </div>

                    <div class="evaluation-section">
                        <h3>{{your_evaluation}}</h3>

                        <div class="scoring-guide">
                            <h4>{{scoring_criteria}}</h4>
                            <ul class="score-criteria-list">
                                <li><strong>1-3:</strong> {{score_1_3}}</li>
                                <li><strong>4-6:</strong> {{score_4_6}}</li>
                                <li><strong>7-8:</strong> {{score_7_8}}</li>
                                <li><strong>9-10:</strong> {{score_9_10}}</li>
                            </ul>
                        </div>

                        <div class="form-group">
                            <label>{{preference_question}}</label>
                            <div class="radio-group">
                                <label class="radio-option">
                                    <input type="radio" name="pref_${task.evaluation_id}"
                                           value="public" onchange="updatePreference('${task.evaluation_id}', 'public')">
                                    {{query_a_better}}
                                </label>
                                <label class="radio-option">
                                    <input type="radio" name="pref_${task.evaluation_id}"
                                           value="personalized" onchange="updatePreference('${task.evaluation_id}', 'personalized')">
                                    {{query_b_better}}
                                </label>
                                <label class="radio-option">
                                    <input type="radio" name="pref_${task.evaluation_id}"
                                           value="tie" onchange="updatePreference('${task.evaluation_id}', 'tie')">
                                    {{tie}}
                                </label>
                            </div>
                        </div>

                        <div class="form-group">
                            <label>{{query_a_score}}</label>
                            <div class="score-input">
                                <input type="range" id="range-public-${task.evaluation_id}" min="1" max="10" value="5"
                                       oninput="syncScoreInputs('${task.evaluation_id}', 'public', this.value)">
                                <input type="number" id="num-public-${task.evaluation_id}" min="1" max="10" value="5"
                                       oninput="syncScoreInputs('${task.evaluation_id}', 'public', this.value)">
                            </div>
                        </div>

                        <div class="form-group">
                            <label>{{query_b_score}}</label>
                            <div class="score-input">
                                <input type="range" id="range-personalized-${task.evaluation_id}" min="1" max="10" value="5"
                                       oninput="syncScoreInputs('${task.evaluation_id}', 'personalized', this.value)">
                                <input type="number" id="num-personalized-${task.evaluation_id}" min="1" max="10" value="5"
                                       oninput="syncScoreInputs('${task.evaluation_id}', 'personalized', this.value)">
                            </div>
                        </div>

                        <div class="form-group">
                            <label>{{reasoning}}</label>
                            <textarea id="reasoning-${task.evaluation_id}" placeholder="{{reasoning_placeholder}}"
                                      onblur="updateReasoning('${task.evaluation_id}', this.value)"></textarea>
                        </div>

                        <div class="llm-scores hidden" id="llm-${task.evaluation_id}">
                            <h4>{{llm_scores_title}}</h4>
                            <p><strong>{{llm_public}}:</strong> ${task.llm_scores.public_score}/10</p>
                            <p><strong>{{llm_personalized}}:</strong> ${task.llm_scores.personalized_score}/10</p>
                            <p><strong>{{llm_prefers}}:</strong> ${task.llm_scores.llm_prefers}</p>
                        </div>
                    </div>
                `;

                container.appendChild(taskDiv);

                // Load saved values
                if (results[task.evaluation_id]) {
                    const result = results[task.evaluation_id];
                    if (result.preferred_query) {
                        taskDiv.querySelector(`input[name="pref_${task.evaluation_id}"][value="${result.preferred_query}"]`).checked = true;
                    }
                    if (result.public_query_score) {
                        taskDiv.querySelector('#range-public-' + task.evaluation_id).value = result.public_query_score;
                        taskDiv.querySelector('#num-public-' + task.evaluation_id).value = result.public_query_score;
                    }
                    if (result.personalized_query_score) {
                        taskDiv.querySelector('#range-personalized-' + task.evaluation_id).value = result.personalized_query_score;
                        taskDiv.querySelector('#num-personalized-' + task.evaluation_id).value = result.personalized_query_score;
                    }
                    if (result.reasoning) {
                        taskDiv.querySelector('textarea').value = result.reasoning;
                    }
                }
            });

            document.getElementById('totalCount').textContent = tasks.length;
        }

        function selectQuery(taskId, queryId) {
            const queryBox = document.getElementById(`query-${queryId}`);
            // Visual feedback only
        }

        function updatePreference(taskId, value) {
            if (!results[taskId]) results[taskId] = {};
            results[taskId].preferred_query = value;
            results[taskId].completed = isTaskComplete(taskId);
            saveToLocalStorage();
            updateProgress();
        }

        function updateScore(taskId, field, value) {
            if (!results[taskId]) results[taskId] = {};
            results[taskId][field] = parseInt(value);
            results[taskId].completed = isTaskComplete(taskId);
            saveToLocalStorage();
            updateProgress();
        }

        function syncScoreInputs(taskId, queryType, value) {
            // Sync range slider and number input
            const rangeInput = document.getElementById(`range-${queryType}-${taskId}`);
            const numInput = document.getElementById(`num-${queryType}-${taskId}`);

            // Update both inputs
            if (rangeInput && numInput) {
                rangeInput.value = value;
                numInput.value = value;
            }

            // Update results and save with correct field names
            if (!results[taskId]) results[taskId] = {};
            if (queryType === 'public') {
                results[taskId].public_query_score = parseInt(value);
            } else if (queryType === 'personalized') {
                results[taskId].personalized_query_score = parseInt(value);
            }
            results[taskId].completed = isTaskComplete(taskId);

            console.log('syncScoreInputs:', { taskId, queryType, value, 'result': results[taskId], 'isComplete': results[taskId].completed });

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
            return result && result.preferred_query && result.public_query_score && result.personalized_query_score;
        }

        function updateProgress() {
            const completed = Object.values(results).filter(r => r.completed).length;
            const total = tasks.length;
            const percentage = Math.round((completed / total) * 100);

            console.log('updateProgress:', { completed, total, percentage, 'results keys': Object.keys(results) });

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
                            statusBadge.textContent = '✓ Completed';
                            statusBadge.className = 'task-status completed';
                        }
                    }
                }
            });
        }

        function saveToFile() {
            // Download backup file with fixed name
            const exportData = tasks.map(task => ({
                ...task,
                human_evaluation: results[task.evaluation_id] || {
                    preferred_query: null,
                    public_query_score: null,
                    personalized_query_score: null,
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
            alert(`✅ 进度已保存！\n\n已完成 ${completedCount}/${tasks.length} 个评估\n\n文件名: human_eval_progress.json`);
        }

        function loadFromFile(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const loadedData = JSON.parse(e.target.result);

                    // Extract results from loaded data
                    const loadedResults = {};
                    let loadedCount = 0;

                    loadedData.forEach(item => {
                        if (item.human_evaluation && item.human_evaluation.preferred_query) {
                            loadedResults[item.evaluation_id] = item.human_evaluation;
                            loadedCount++;
                        }
                    });

                    // Replace existing results
                    results = loadedResults;

                    // Save to localStorage
                    saveToLocalStorage();

                    // Re-render tasks to load saved values
                    renderTasks();
                    updateProgress();

                    alert(`✅ 进度已加载！\n\n成功恢复 ${loadedCount} 个评估`);
                } catch (error) {
                    alert('❌ 加载文件失败: ' + error.message);
                }
            };
            reader.readAsText(file);

            // Reset file input
            event.target.value = '';
        }

        function clearAll() {
            if (confirm('⚠️ Are you sure you want to clear all progress? This cannot be undone.')) {
                localStorage.removeItem('humanEvalResults');
                results = {};
                location.reload();
            }
        }

        function exportResults() {
            const exportData = tasks.map(task => ({
                ...task,
                human_evaluation: results[task.evaluation_id] || {
                    preferred_query: null,
                    public_query_score: null,
                    personalized_query_score: null,
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
    </script>
</body>
</html>"""

    # Replace placeholder with actual tasks data
    tasks_json = json.dumps(tasks, ensure_ascii=False, indent=2)
    html_content = html_template.replace('__TASKS_DATA__', tasks_json)

    # Replace language strings
    for key, value in strings.items():
        html_content = html_content.replace(f'{{{{{key}}}}}', value)

    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    log_with_timestamp(f"HTML interface saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate human evaluation tasks and HTML interface"
    )
    parser.add_argument(
        "--stage10-dir",
        default="/home/wlia0047/wenyu/result/user_profile/10_evaluation",
        help="Directory containing Stage 10 evaluation results"
    )
    parser.add_argument(
        "--stage9-dir",
        default="/home/wlia0047/wenyu/result/user_profile/09_targeted_noisy_query",
        help="Directory containing Stage 9 noisy query results"
    )
    parser.add_argument(
        "--persona-dir",
        default="/home/wlia0047/wenyu/result/user_profile/03_persona/results",
        help="Directory containing user persona files"
    )
    parser.add_argument(
        "--output-dir",
        default="/home/wlia0047/wenyu/result/user_profile/11_human_evaluation/tasks",
        help="Output directory for evaluation tasks and HTML interface"
    )
    parser.add_argument(
        "--language",
        default="en",
        choices=['en', 'zh'],
        help="Interface language: en for English, zh for Chinese"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all data sources
    evaluations = load_stage10_evaluations(args.stage10_dir)
    noisy_queries = load_stage9_noisy_queries(args.stage9_dir)
    personas = load_stage3_personas(args.persona_dir)

    # Generate evaluation tasks
    tasks, stats = generate_evaluation_tasks(evaluations, noisy_queries, personas)

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
    log_with_timestamp("GENERATION COMPLETE")
    log_with_timestamp("="*70)
    log_with_timestamp(f"Total evaluation tasks: {stats['total_query_pairs']}")
    log_with_timestamp(f"Total users: {stats['total_users']}")
    log_with_timestamp(f"\nOutput files:")
    log_with_timestamp(f"  1. {tasks_json_path}")
    log_with_timestamp(f"  2. {html_path}")
    log_with_timestamp(f"\nNext steps:")
    log_with_timestamp(f"  1. Open {html_path} in your browser")
    log_with_timestamp(f"  2. Complete the evaluation for all {stats['total_query_pairs']} query pairs")
    log_with_timestamp(f"  3. Download the results as JSON")
    log_with_timestamp(f"  4. Run 11_compute_alignment_metrics.py to calculate alignment metrics")
    log_with_timestamp("="*70)


if __name__ == "__main__":
    main()
