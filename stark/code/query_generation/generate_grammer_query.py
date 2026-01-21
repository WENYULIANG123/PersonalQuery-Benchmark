import json
import os
import sys
import csv
import time
from typing import List, Dict, Any

# 导入LLM模型接口
CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CODE_DIR not in sys.path:
    sys.path.append(CODE_DIR)
from model import get_gm_model

def assess_grammar_dna_from_analysis_file(analysis_file_path):
    """
    从用户语法分析JSON文件中加载和解析用户的语法DNA。

    Args:
        analysis_file_path (str): 用户语法分析JSON文件的路径

    Returns:
        dict: 包含用户语法DNA信息的字典，包含以下键：
            - user_id: 用户ID
            - fingerprint: 用户语法指纹数据
            - padding_phrases: 填充词列表
            - interruption_types: 打断类型列表
            - preferred_generic_nouns: 偏好的通用名词列表
            - analysis_summary: 分析摘要
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(analysis_file_path):
            raise FileNotFoundError(f"User grammar analysis file not found: {analysis_file_path}")

        # 读取并解析JSON文件
        with open(analysis_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 验证必要的字段
        if 'user_id' not in data:
            raise ValueError("Missing 'user_id' field in grammar analysis file")
        if 'fingerprint' not in data:
            raise ValueError("Missing 'fingerprint' field in grammar analysis file")

        fingerprint = data['fingerprint']

        # 提取关键的用户语法特征
        user_grammar_dna = {
            'user_id': data['user_id'],
            'fingerprint': fingerprint,
            'padding_phrases': [],
            'interruption_types': [],
            'preferred_generic_nouns': [],
            'analysis_summary': data.get('analysis_summary', {})
        }

        # 提取填充词（padding phrases）
        if 'syntactic' in fingerprint and 'padding_phrases' in fingerprint['syntactic']:
            user_grammar_dna['padding_phrases'] = [
                item['phrase'] for item in fingerprint['syntactic']['padding_phrases']
                if item['frequency'] > 0  # 只包含使用过的填充词
            ]

        # 提取打断类型（interruption types）
        if 'syntactic' in fingerprint and 'interruption_types' in fingerprint['syntactic']:
            user_grammar_dna['interruption_types'] = [
                item['type'] for item in fingerprint['syntactic']['interruption_types']
                if item['frequency'] > 0  # 只包含使用过的打断类型
            ]

        # 提取偏好的通用名词（preferred generic nouns）
        if 'lexical' in fingerprint and 'preferred_generic_nouns' in fingerprint['lexical']:
            user_grammar_dna['preferred_generic_nouns'] = [
                item['noun'] for item in fingerprint['lexical']['preferred_generic_nouns']
                if item['frequency'] > 0  # 只包含使用过的名词
            ]

        print(f"✅ Successfully loaded user grammar DNA for user: {user_grammar_dna['user_id']}")
        print(f"📊 Padding phrases: {len(user_grammar_dna['padding_phrases'])} types")
        print(f"📊 Interruption types: {len(user_grammar_dna['interruption_types'])} types")
        print(f"📊 Generic nouns: {len(user_grammar_dna['preferred_generic_nouns'])} types")

        return user_grammar_dna

    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        raise
    except KeyError as e:
        print(f"❌ Missing required field in grammar analysis file: {e}")
        raise
    except Exception as e:
        print(f"❌ Unexpected error loading grammar analysis file: {e}")
        raise


def analyze_query_syntax_for_retrieval_degradation(query_text: str, llm_model, query_id: int = None, verbose: bool = True) -> Dict[str, Any]:
    """
    分析查询的语法结构，生成降低检索性能的改写建议。

    Args:
        query_text (str): 原始查询文本
        llm_model: LLM模型实例
        query_id (int): 查询ID（用于日志）
        verbose (bool): 是否打印详细日志

    Returns:
        dict: 包含改写建议的字典
    """
    try:
        if verbose:
            print(f"🤖 Analyzing syntax for query {query_id}: '{query_text}'")

        # 构造prompt - 专门针对检索性能降低
        prompt = f"""Analyze this search query for syntactic weaknesses affecting retrieval performance on BM25, ANCE, and ColBERTv2.

Query: "{query_text}"

Provide a concise analysis in this format:

## Query Strengths
[List 2-3 key grammatical advantages]

## Key Weaknesses
[Numbered list of 4-6 specific vulnerabilities]

## Impact Summary
| Weakness | BM25 Impact | ANCE Impact | ColBERTv2 Impact |
|----------|-------------|-------------|------------------|
| [Weakness 1] | [1-2 sentence explanation] | [1-2 sentence explanation] | [1-2 sentence explanation] |
| [Weakness 2] | ... | ... | ... |

Keep explanations technical but concise (max 2 sentences per cell). Focus on WHY each weakness hurts performance."""

        # 调用LLM
        start_time = time.time()
        response = llm_model.invoke(prompt)
        response_time = time.time() - start_time

        if verbose:
            print(f"{response_time:.2f}")
            print(f"📝 Response length: {len(response.content)} characters")
            print(f"📝 LLM Response content:")
            print(response.content)
            print("="*80)

        # 解析响应并提取结构化信息
        try:
            # 尝试解析LLM响应，提取关键信息
            parsed_analysis = parse_llm_analysis_response(response.content)

            analysis_result = {
                'query_id': query_id,
                'original_query': query_text,
                'llm_response': response.content,
                'parsed_analysis': parsed_analysis,
                'response_time': response_time,
                'timestamp': time.time(),
                'success': True
            }
        except Exception as parse_error:
            print(f"⚠️ Failed to parse LLM response: {parse_error}")
            analysis_result = {
                'query_id': query_id,
                'original_query': query_text,
                'llm_response': response.content,
                'parsed_analysis': None,
                'response_time': response_time,
                'timestamp': time.time(),
                'success': True  # 仍然标记为成功，因为LLM调用成功了
            }

        return analysis_result

    except Exception as e:
        print(f"❌ Error analyzing query {query_id}: {e}")
        return {
            'query_id': query_id,
            'original_query': query_text,
            'error': str(e),
            'success': False,
            'timestamp': time.time()
        }


def parse_llm_analysis_response(llm_response):
    """
    解析LLM紧凑弱点分析响应的内容，提取结构化信息。

    Args:
        llm_response (str): LLM的原始响应

    Returns:
        dict: 解析后的结构化数据
    """
    try:
        lines = llm_response.strip().split('\n')
        parsed = {
            'query_strengths': [],
            'weaknesses': [],
            'impact_table': []
        }

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测章节标题
            if line.startswith('## Query Strengths'):
                current_section = 'strengths'
            elif line.startswith('## Key Weaknesses'):
                current_section = 'weaknesses'
            elif line.startswith('## Impact Summary'):
                current_section = 'impact_table'
            elif current_section == 'strengths' and line.startswith('- '):
                # 提取语法优势
                strength = line.replace('- ', '').strip()
                if strength:
                    parsed['query_strengths'].append(strength)
            elif current_section == 'weaknesses' and line.strip():
                # 提取弱点（跳过标题行）
                if not line.startswith('##') and not line.startswith('['):
                    # 移除编号前缀 (如 "1. ", "[1] " 等)
                    weakness = line.lstrip('0123456789.[] ').strip()
                    if weakness and len(weakness) > 5:  # 避免空行或过短文本
                        parsed['weaknesses'].append(weakness)
            elif current_section == 'impact_table' and '|' in line and '---' not in line:
                # 解析表格行（跳过分隔线）
                if 'Weakness' not in line and 'BM25' not in line:  # 跳过表头
                    parts = [part.strip() for part in line.split('|') if part.strip()]
                    if len(parts) >= 4:
                        weakness_name = parts[0]
                        bm25_impact = parts[1]
                        ance_impact = parts[2]
                        colbertv2_impact = parts[3]

                        parsed['impact_table'].append({
                            'weakness': weakness_name,
                            'bm25_impact': bm25_impact,
                            'ance_impact': ance_impact,
                            'colbertv2_impact': colbertv2_impact
                        })

        return parsed

    except Exception as e:
        print(f"⚠️ Error parsing LLM response: {e}")
        return {
            'raw_response': llm_response,
            'parse_error': str(e)
        }


def generate_user_aware_query_rewrite(original_query: str, retrieval_analysis: dict, user_grammar_dna: dict, llm_model, query_id: int = None, verbose: bool = True) -> Dict[str, Any]:
    """
    基于检索分析和用户画像生成个性化的查询改写。

    Args:
        original_query (str): 原始查询
        retrieval_analysis (dict): 第一步的检索性能分析结果
        user_grammar_dna (dict): 用户语法DNA
        llm_model: LLM模型实例
        query_id (int): 查询ID
        verbose (bool): 是否打印详细日志

    Returns:
        dict: 包含个性化改写结果的字典
    """
    try:
        if verbose:
            print(f"🎨 Generating user-aware rewrite for query {query_id}")

        # 提取用户画像的关键特征
        padding_phrases = user_grammar_dna.get('padding_phrases', [])
        generic_nouns = user_grammar_dna.get('preferred_generic_nouns', [])
        interruption_types = user_grammar_dna.get('interruption_types', [])

        # 构造结合用户画像的prompt
        prompt = f"""Based on the retrieval performance analysis and user grammar profile, create a personalized query rewrite.

**Original Query:** "{original_query}"

**Retrieval Analysis:** {retrieval_analysis.get('llm_response', 'N/A')}

**User Grammar Profile:**
- Preferred padding phrases: {', '.join(padding_phrases)}
- Preferred generic nouns: {', '.join(generic_nouns)}
- Common interruption types: {', '.join(interruption_types)}

**Task:** Create a rewritten query that:
1. Incorporates the user's preferred linguistic patterns (padding phrases, generic nouns)
2. Maintains the core meaning of the original query
3. Follows the user's typical interruption and sentence structure habits
4. May include elements that could affect retrieval performance (based on the analysis)

**Output Format:**
1. **User-Aware Rewrite**: The rewritten query
2. **Incorporated User Patterns**: Which user patterns were used and why
3. **Expected User Reception**: How well this matches the user's writing style
4. **Retrieval Impact Assessment**: How this rewrite might affect BM25/ANCE/ColBERTv2 performance

Make the rewrite sound natural while incorporating the user's linguistic fingerprint."""

        # 调用LLM
        start_time = time.time()
        response = llm_model.invoke(prompt)
        response_time = time.time() - start_time

        if verbose:
            print(f"{response_time:.2f}")
            print(f"📝 Response length: {len(response.content)} characters")
            print(f"📝 LLM Response content:")
            print(response.content)
            print("="*80)

        # 解析响应
        result = {
            'query_id': query_id,
            'original_query': original_query,
            'user_grammar_dna': user_grammar_dna,
            'retrieval_analysis': retrieval_analysis,
            'llm_response': response.content,
            'response_time': response_time,
            'timestamp': time.time(),
            'success': True
        }

        return result

    except Exception as e:
        print(f"❌ Error in user-aware rewrite for query {query_id}: {e}")
        return {
            'query_id': query_id,
            'original_query': original_query,
            'error': str(e),
            'success': False,
            'timestamp': time.time()
        }


def two_stage_query_rewriting_workflow(queries: List[Dict], user_grammar_dna: dict, llm_model, max_queries: int = None, verbose: bool = True) -> List[Dict]:
    """
    两阶段查询改写工作流：先分析检索影响，再结合用户画像改写。

    Args:
        queries: 查询列表
        user_grammar_dna: 用户语法DNA
        llm_model: LLM模型实例
        max_queries: 最大处理查询数量
        verbose: 是否打印详细日志

    Returns:
        list: 完整的改写结果列表
    """
    results = []
    total_queries = len(queries) if max_queries is None else min(max_queries, len(queries))

    print(f"\n{'='*100}")
    print(f"🚀 STARTING TWO-STAGE QUERY REWRITING WORKFLOW")
    print(f"{'='*100}")
    print(f"📊 Total queries to process: {total_queries}")
    print(f"🎯 Stage 1: Retrieval Impact Analysis")
    print(f"🎯 Stage 2: User-Aware Personalized Rewriting")
    print(f"🤖 Using LLM model: {type(llm_model).__name__}")

    start_time = time.time()
    stage1_success = 0
    stage2_success = 0

    for i, query in enumerate(queries[:total_queries]):
        query_start_time = time.time()

        if verbose:
            print(f"\n{'='*80}")
            print(f"🔍 Processing Query {i+1}/{total_queries} (ID: {query['id']})")
            print(f"{'='*80}")

        # Stage 1: 检索性能分析
        if verbose:
            print(f"📈 Stage 1: Analyzing retrieval impact...")

        retrieval_analysis = analyze_query_syntax_for_retrieval_degradation(
            query_text=query['query'],
            llm_model=llm_model,
            query_id=query['id'],
            verbose=verbose
        )

        if retrieval_analysis['success']:
            stage1_success += 1

            # Stage 2: 用户画像个性化改写
            if verbose:
                print(f"🎨 Stage 2: Generating user-aware rewrite...")

            user_rewrite = generate_user_aware_query_rewrite(
                original_query=query['query'],
                retrieval_analysis=retrieval_analysis,
                user_grammar_dna=user_grammar_dna,
                llm_model=llm_model,
                query_id=query['id'],
                verbose=verbose
            )

            if user_rewrite['success']:
                stage2_success += 1
            else:
                user_rewrite = None

        # 组合结果
        combined_result = {
            'query_id': query['id'],
            'original_query': query['query'],
            'stage1_retrieval_analysis': retrieval_analysis,
            'stage2_user_rewrite': user_rewrite,
            'processing_time': time.time() - query_start_time,
            'success': retrieval_analysis['success'] and (user_rewrite['success'] if user_rewrite else False)
        }

        results.append(combined_result)

        # 进度报告
        if (i + 1) % 2 == 0:  # 每2个查询报告一次进度
            elapsed = time.time() - start_time
            print(f"\n📈 Progress: {i+1}/{total_queries} queries completed")
            print(f"   Stage 1 Success: {stage1_success}/{i+1}")
            print(f"   Stage 2 Success: {stage2_success}/{stage1_success}")
            print(f"   Elapsed: {elapsed:.2f}s")

    # 最终统计
    total_time = time.time() - start_time
    print(f"\n{'='*100}")
    print(f"🏁 TWO-STAGE WORKFLOW COMPLETED")
    print(f"{'='*100}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Queries processed: {total_queries}")
    print(f"📈 Stage 1 (Retrieval Analysis): {stage1_success}/{total_queries} successful")
    print(f"🎨 Stage 2 (User-Aware Rewriting): {stage2_success}/{stage1_success} successful")
    print(".3f")
    if stage1_success > 0:
        print(".2f")

    return results


def batch_analyze_queries_for_retrieval_degradation(queries: List[Dict], llm_model, max_queries: int = None, verbose: bool = True) -> List[Dict]:
    """
    批量分析所有查询的语法改写建议。

    Args:
        queries: 查询列表
        llm_model: LLM模型实例
        max_queries: 最大处理查询数量（None表示全部）
        verbose: 是否打印详细日志

    Returns:
        list: 分析结果列表
    """
    results = []
    total_queries = len(queries) if max_queries is None else min(max_queries, len(queries))

    print(f"\n{'='*80}")
    print(f"🚀 STARTING BATCH SYNTAX ANALYSIS FOR RETRIEVAL DEGRADATION")
    print(f"{'='*80}")
    print(f"📊 Total queries to analyze: {total_queries}")
    print(f"🎯 Target: Reduce BM25/ANCE/ColBERTv2 retrieval performance")
    print(f"🤖 Using LLM model: {type(llm_model).__name__}")

    start_time = time.time()
    successful_analyses = 0
    failed_analyses = 0

    for i, query in enumerate(queries[:total_queries]):
        query_start_time = time.time()

        if verbose:
            print(f"\n{'-'*60}")
            print(f"🔍 Processing Query {i+1}/{total_queries} (ID: {query['id']})")
            print(f"{'-'*60}")

        # 分析查询
        result = analyze_query_syntax_for_retrieval_degradation(
            query_text=query['query'],
            llm_model=llm_model,
            query_id=query['id'],
            verbose=verbose
        )

        # 记录结果
        results.append(result)

        if result['success']:
            successful_analyses += 1
        else:
            failed_analyses += 1

        query_time = time.time() - query_start_time
        if verbose:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {query_time:.2f}s")
        # 每处理10个查询显示一次进度
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"\n📈 Progress: {i+1}/{total_queries} queries processed")
            print(f"⏱️ Elapsed: {elapsed:.2f}s")
    # 最终统计
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"🏁 BATCH ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Queries processed: {total_queries}")
    print(f"✅ Successful analyses: {successful_analyses}")
    print(f"❌ Failed analyses: {failed_analyses}")
    print(".3f")
    if successful_analyses > 0:
        print(".2f")
    return results


def load_human_generated_queries(csv_file_path):
    """
    从CSV文件中加载人工生成的查询数据。

    Args:
        csv_file_path (str): CSV文件路径

    Returns:
        list: 包含查询数据的字典列表，每个字典包含：
            - id: 查询ID
            - query: 查询文本
            - answer_ids: 相关答案ID列表
            - answer_ids_source: 来源答案ID列表
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"Query CSV file not found: {csv_file_path}")

        queries = []

        # 读取CSV文件
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            # 使用csv.DictReader自动处理字段名
            reader = csv.DictReader(csvfile)

            for row_num, row in enumerate(reader, 1):
                try:
                    # 验证必需字段
                    if 'id' not in row or 'query' not in row:
                        raise ValueError(f"Missing required fields 'id' or 'query' in row {row_num}")

                    # 解析answer_ids (JSON格式的列表)
                    answer_ids = []
                    if row.get('answer_ids'):
                        try:
                            # 移除可能的引号并解析JSON
                            answer_ids_str = row['answer_ids'].strip('"')
                            answer_ids = json.loads(answer_ids_str)
                        except json.JSONDecodeError:
                            print(f"⚠️ Warning: Could not parse answer_ids in row {row_num}: {row['answer_ids']}")

                    # 解析answer_ids_source (JSON格式的列表)
                    answer_ids_source = []
                    if row.get('answer_ids_source'):
                        try:
                            # 移除可能的引号并解析JSON
                            source_str = row['answer_ids_source'].strip('"')
                            answer_ids_source = json.loads(source_str)
                        except json.JSONDecodeError:
                            print(f"⚠️ Warning: Could not parse answer_ids_source in row {row_num}: {row['answer_ids_source']}")

                    # 创建查询字典
                    query_data = {
                        'id': int(row['id']),
                        'query': row['query'].strip(),
                        'answer_ids': answer_ids,
                        'answer_ids_source': answer_ids_source,
                        'row_number': row_num
                    }

                    queries.append(query_data)

                except Exception as e:
                    print(f"❌ Error processing row {row_num}: {e}")
                    continue

        print(f"✅ Successfully loaded {len(queries)} queries from CSV file")
        print(f"📊 Total rows processed: {row_num}")

        # 统计信息
        total_answer_ids = sum(len(q['answer_ids']) for q in queries)
        total_source_ids = sum(len(q['answer_ids_source']) for q in queries)
        avg_query_length = sum(len(q['query'].split()) for q in queries) / len(queries) if queries else 0

        print(f"📊 Average query length: {avg_query_length:.1f} words")
        print(f"📊 Total answer IDs: {total_answer_ids}")
        print(f"📊 Total source answer IDs: {total_source_ids}")

        return queries

    except Exception as e:
        print(f"❌ Unexpected error loading CSV file: {e}")
        raise


def print_query_details(queries, max_display=10):
    """打印查询数据的详细信息"""
    print(f"\n{'='*100}")
    print(f"📋 QUERY DATA DETAILS (showing first {min(max_display, len(queries))} queries)")
    print(f"{'='*100}")

    for i, query in enumerate(queries[:max_display]):
        print(f"\n🔍 Query #{query['id']} (Row {query['row_number']}):")
        print(f"   📝 Text: {query['query']}")
        print(f"   🎯 Answer IDs: {len(query['answer_ids'])} items - {query['answer_ids'][:5]}{'...' if len(query['answer_ids']) > 5 else ''}")
        print(f"   📍 Source IDs: {len(query['answer_ids_source'])} items - {query['answer_ids_source']}")

        # 分析查询长度
        word_count = len(query['query'].split())
        char_count = len(query['query'])
        print(f"   📊 Length: {word_count} words, {char_count} characters")

    if len(queries) > max_display:
        print(f"\n... and {len(queries) - max_display} more queries")

    print(f"\n{'='*100}")


def print_user_profile_details(user_grammar_dna):
    """详细打印用户画像的所有内容"""
    fingerprint = user_grammar_dna['fingerprint']

    print(f"\n{'='*100}")
    print(f"📋 DETAILED USER PROFILE CONTENTS")
    print(f"{'='*100}")

    # 基本信息
    print(f"👤 User ID: {user_grammar_dna['user_id']}")

    # 语法特征 (Syntactic)
    if 'syntactic' in fingerprint:
        print(f"\n🔤 SYNTACTIC FEATURES:")
        print(f"{'-'*30}")

        # Padding Phrases
        if 'padding_phrases' in fingerprint['syntactic']:
            print(f"📝 Padding Phrases ({len(fingerprint['syntactic']['padding_phrases'])} types):")
            for item in fingerprint['syntactic']['padding_phrases']:
                print(f"   • '{item['phrase']}' (frequency: {item['frequency']})")

        # Interruption Types
        if 'interruption_types' in fingerprint['syntactic']:
            print(f"\n🔄 Interruption Types ({len(fingerprint['syntactic']['interruption_types'])} types):")
            for item in fingerprint['syntactic']['interruption_types']:
                print(f"   • {item['type']} (frequency: {item['frequency']})")

        # Conditional Usage Levels
        if 'conditional_usage_levels' in fingerprint['syntactic']:
            print(f"\n🔀 Conditional Usage Levels ({len(fingerprint['syntactic']['conditional_usage_levels'])} levels):")
            for item in fingerprint['syntactic']['conditional_usage_levels']:
                print(f"   • {item['level']} (frequency: {item['frequency']})")

    # 词汇特征 (Lexical)
    if 'lexical' in fingerprint:
        print(f"\n📊 LEXICAL FEATURES:")
        print(f"{'-'*30}")

        # Circumlocution Habits
        if 'circumlocution_habits' in fingerprint['lexical']:
            print(f"🔄 Circumlocution Habits ({len(fingerprint['lexical']['circumlocution_habits'])} types):")
            for item in fingerprint['lexical']['circumlocution_habits']:
                print(f"   • {item['type']} (frequency: {item['frequency']})")

        # Preferred Generic Nouns
        if 'preferred_generic_nouns' in fingerprint['lexical']:
            print(f"\n📊 Preferred Generic Nouns ({len(fingerprint['lexical']['preferred_generic_nouns'])} nouns):")
            for item in fingerprint['lexical']['preferred_generic_nouns']:
                print(f"   • '{item['noun']}' (frequency: {item['frequency']})")

    # 逻辑特征 (Logic)
    if 'logic' in fingerprint:
        print(f"\n🧠 LOGIC FEATURES:")
        print(f"{'-'*30}")

        # Past Reference Habits
        if 'past_reference_habits' in fingerprint['logic']:
            print(f"⏰ Past Reference Habits ({len(fingerprint['logic']['past_reference_habits'])} levels):")
            for item in fingerprint['logic']['past_reference_habits']:
                print(f"   • {item['level']} (frequency: {item['frequency']})")

        # Negation Styles
        if 'negation_styles' in fingerprint['logic']:
            print(f"\n❌ Negation Styles ({len(fingerprint['logic']['negation_styles'])} styles):")
            for item in fingerprint['logic']['negation_styles']:
                print(f"   • {item['type']} (frequency: {item['frequency']})")

    # 上下文洞察 (Contextual Insights)
    if 'contextual_insights' in fingerprint:
        print(f"\n💡 CONTEXTUAL INSIGHTS:")
        print(f"{'-'*30}")

        insights = fingerprint['contextual_insights']
        if 'syntactic' in insights:
            print("🔤 Syntactic Insights:")
            for key, value in insights['syntactic'].items():
                print(f"   • {key}: {value}")

        if 'lexical' in insights:
            print("\n📊 Lexical Insights:")
            for key, value in insights['lexical'].items():
                print(f"   • {key}: {value}")

        if 'logic' in insights:
            print("\n🧠 Logic Insights:")
            for key, value in insights['logic'].items():
                print(f"   • {key}: {value}")

    # 风格总结和分析摘要
    if 'style_summary' in fingerprint:
        print(f"\n✨ STYLE SUMMARY:")
        print(f"{'-'*30}")
        print(f"{fingerprint['style_summary']}")

    if 'analysis_summary' in fingerprint:
        print(f"\n📈 ANALYSIS SUMMARY:")
        print(f"{'-'*30}")
        summary = fingerprint['analysis_summary']
        if 'total_sentences_analyzed' in summary:
            print(f"📊 Total Sentences Analyzed: {summary['total_sentences_analyzed']}")

        if 'dimension_percentages' in summary:
            print("📊 Dimension Percentages:")
            percentages = summary['dimension_percentages']
            for key, value in percentages.items():
                print(f"   • {key}: {value:.2f}%")

        if 'raw_counts' in summary:
            print("\n📊 Raw Counts:")
            raw_counts = summary['raw_counts']
            for key, value in raw_counts.items():
                print(f"   • {key}: {value}")

    print(f"\n{'='*100}")


def main():
    """主函数 - 演示用户画像和查询数据加载功能"""
    import time

    print("🚀 Starting main function...")
    start_time = time.time()
    print(f"🚀 [{time.time() - start_time:.1f}s] Starting User Profile & Query Data Loading Demo...")

    # 文件路径
    analysis_file_path = "/home/wlia0047/ar57_scratch/wenyu/amazon_review_grammer_analysis.json"
    csv_file_path = "/home/wlia0047/ar57/wenyu/stark/data/stark_qa_human_generated_eval.csv"

    try:
        # 第一步：加载用户语法DNA
        print(f"\n{'='*60}")
        print(f"🎯 STEP 1: Loading User Grammar DNA")
        print(f"{'='*60}")
        print(f"📋 [{time.time() - start_time:.1f}s] Loading user grammar analysis file...")
        user_grammar_dna = assess_grammar_dna_from_analysis_file(analysis_file_path)

        # 第二步：加载查询数据
        print(f"\n{'='*60}")
        print(f"🎯 STEP 2: Loading Human Generated Queries")
        print(f"{'='*60}")
        print(f"📄 [{time.time() - start_time:.1f}s] Loading query CSV file...")
        queries = load_human_generated_queries(csv_file_path)

        # 显示数据概览
        print(f"\n{'='*80}")
        print(f"📊 DATA LOADING SUMMARY")
        print(f"{'='*80}")
        print(f"👤 User Profile: {user_grammar_dna['user_id']}")
        print(f"📝 Queries Loaded: {len(queries)} items")
        print(f"📈 Analysis Sentences: {user_grammar_dna['fingerprint']['analysis_summary']['total_sentences_analyzed']}")

        # 详细打印前几个查询
        print_query_details(queries, max_display=3)

        # 第三步：两阶段查询改写工作流
        print(f"\n{'='*60}")
        print(f"🎯 STEP 3: Two-Stage Query Rewriting Workflow")
        print(f"{'='*60}")
        print(f"📈 Stage 3.1: Retrieval Impact Analysis")
        print(f"🎨 Stage 3.2: User-Aware Personalized Rewriting")

        # 初始化LLM模型
        print(f"🤖 [{time.time() - start_time:.1f}s] Initializing LLM model...")
        try:
            llm_model = get_gm_model()
            print(f"✅ [{time.time() - start_time:.1f}s] LLM model initialized successfully")
        except Exception as e:
            print(f"❌ [{time.time() - start_time:.1f}s] Failed to initialize LLM: {e}")
            print(f"⚠️  Continuing with data loading only...")
            llm_model = None

        # 执行第一阶段检索影响分析（只分析前3个查询用于演示）
        if llm_model:
            print(f"🔬 [{time.time() - start_time:.1f}s] Starting retrieval impact analysis...")
            analysis_results = batch_analyze_queries_for_retrieval_degradation(
                queries=queries,
                llm_model=llm_model,
                max_queries=3,  # 演示用，只处理前3个
                verbose=True
            )

            print(f"\n📋 [{time.time() - start_time:.1f}s] Retrieval impact analysis completed for {len(analysis_results)} queries")

            # 展示分析结果
            display_analysis_results(analysis_results, max_display=3)

            # 保存分析结果
            output_file = "/home/wlia0047/ar57_scratch/wenyu/syntax_weakness_analysis_results.json"
            save_analysis_results_to_file(analysis_results, output_file)

        else:
            analysis_results = []
            print(f"⚠️  Skipping analysis due to LLM initialization failure")

        # 最终总结
        print(f"\n{'='*100}")
        print(f"🎉 COMPLETE ANALYSIS EXECUTION SUMMARY")
        print(f"{'='*100}")
        print(f"✅ Step 1 - User Profile: LOADED ({user_grammar_dna['user_id']})")
        print(f"✅ Step 2 - Queries: LOADED ({len(queries)} items)")
        if analysis_results:
            successful_analyses = sum(1 for r in analysis_results if r['success'])
            print(f"✅ Step 3 - Retrieval Impact Analysis: COMPLETED")
            print(f"   📈 Successful Analyses: {successful_analyses}/{len(analysis_results)}")
        else:
            print(f"❌ Step 3 - Retrieval Impact Analysis: SKIPPED (LLM unavailable)")

        print(f"\n⏱️  Total execution time: {time.time() - start_time:.2f} seconds")
        print(f"📋 Grammar file: {analysis_file_path}")
        print(f"📄 Query file: {csv_file_path}")

        # 返回完整结果
        result = {
            'user_grammar_dna': user_grammar_dna,
            'queries': queries,
            'analysis_results': analysis_results,
            'metadata': {
                'grammar_file': analysis_file_path,
                'query_file': csv_file_path,
                'queries_loaded': len(queries),
                'analyses_completed': len(analysis_results),
                'user_id': user_grammar_dna['user_id'],
                'execution_time': time.time() - start_time
            }
        }

        return result

    except Exception as e:
        print(f"❌ [{time.time() - start_time:.1f}s] Failed to load data: {e}")
        sys.exit(1)


def display_analysis_results(analysis_results, max_display=5):
    """
    格式化展示紧凑的分析结果。

    Args:
        analysis_results: 分析结果列表
        max_display: 最多显示多少个结果
    """
    print(f"\n{'='*80}")
    print(f"📊 SYNTAX WEAKNESS ANALYSIS")
    print(f"{'='*80}")

    successful_analyses = [r for r in analysis_results if r.get('success', False)]
    print(f"📈 Total: {len(analysis_results)} | ✅ Success: {len(successful_analyses)}")

    for i, result in enumerate(successful_analyses[:max_display]):
        print(f"\n{'─'*70}")
        print(f"🔍 Query {i+1} (ID: {result.get('query_id', 'N/A')})")
        print(f"📝 {result.get('original_query', 'N/A')}")

        parsed = result.get('parsed_analysis', {})
        if parsed:
            # 语法优势
            if parsed.get('query_strengths'):
                print(f"\n💪 Strengths:")
                for strength in parsed['query_strengths'][:2]:  # 最多显示2个
                    print(f"   • {strength}")

            # 关键弱点
            if parsed.get('weaknesses'):
                print(f"\n⚠️ Weaknesses ({len(parsed['weaknesses'])}):")
                for j, weakness in enumerate(parsed['weaknesses'], 1):
                    print(f"   {j}. {weakness}")

            # 影响表格
            if parsed.get('impact_table'):
                print(f"\n📋 Impact Summary:")
                print(f"   {'Weakness':<25} {'BM25':<8} {'ANCE':<8} {'ColBERTv2'}")
                print(f"   {'─'*25} {'─'*8} {'─'*8} {'─'*10}")
                for row in parsed['impact_table'][:4]:  # 最多显示4个
                    weakness_short = row['weakness'][:24] + "..." if len(row['weakness']) > 24 else row['weakness']
                    bm25_short = row['bm25_impact'][:7] + "..." if len(row['bm25_impact']) > 7 else row['bm25_impact']
                    ance_short = row['ance_impact'][:7] + "..." if len(row['ance_impact']) > 7 else row['ance_impact']
                    colbert_short = row['colbertv2_impact'][:9] + "..." if len(row['colbertv2_impact']) > 9 else row['colbertv2_impact']
                    print(f"   {weakness_short:<25} {bm25_short:<8} {ance_short:<8} {colbert_short}")

        print(f"\n⏱️ {result.get('response_time', 0):.2f}s")

    if len(successful_analyses) > max_display:
        print(f"\n... and {len(successful_analyses) - max_display} more analyses")

    print(f"\n{'='*80}")


def save_analysis_results_to_file(analysis_results, output_file_path):
    """
    将分析结果保存为JSON文件。

    Args:
        analysis_results: 分析结果列表
        output_file_path: 输出文件路径
    """
    try:
        # 准备保存的数据结构
        save_data = {
            'metadata': {
                'total_analyses': len(analysis_results),
                'successful_analyses': sum(1 for r in analysis_results if r.get('success', False)),
                'timestamp': int(time.time()),
                'analysis_type': 'syntax_weakness_analysis'
            },
            'results': analysis_results
        }

        # 保存为JSON文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        print(f"💾 Analysis results saved to: {output_file_path}")
        print(f"📊 Total results: {save_data['metadata']['total_analyses']}")
        print(f"✅ Successful: {save_data['metadata']['successful_analyses']}")

    except Exception as e:
        print(f"❌ Error saving analysis results: {e}")


if __name__ == "__main__":
    main()
