#!/usr/bin/env python3
"""
Evaluate Query-Persona Matching with UNIQUE Personas (V2 - Fixed)
- Focus on NEED matching, not language style
- Score range: 1-10
"""

import json
import os
import sys
import argparse
import time
import random
import re
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def call_llm_api_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            from llm_client import LLMClient
            client = LLMClient()
            response = client.call(prompt, max_tokens=512)
            if response:
                return response
        except Exception as e:
            if "429" in str(e):
                wait_time = (2 ** attempt) + random.random()
                time.sleep(wait_time)
            else:
                log_with_timestamp(f"LLM error: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    return None

def parse_llm_json_response(llm_response):
    try:
        json_content = llm_response
        if "```json" in llm_response:
            json_content = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL).group(1)
        elif "```" in llm_response:
            json_content = re.search(r'```\s*(.*?)\s*```', llm_response, re.DOTALL).group(1)
        else:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
        return json.loads(json_content)
    except:
        return None

def evaluate_query_persona_match(query, persona, query_type):
    """
    Evaluate how well a query matches a user persona.
    Score range: 1-10
    """
    prompt = f"""You are an expert evaluator. Assess how well a search query aligns with a specific user's persona.

**User Persona:**
{persona}

**Search Query ({query_type}):**
"{query}"

**Task:**
Evaluate whether this query reflects the user's UNIQUE PREFERENCES, INTERESTS, and SHOPPING INTENT.

**Key Principle:**
Evaluate NEED matching, not LANGUAGE STYLE. A query can use technical terms OR everyday language - what matters is whether it captures the user's specific requirements.

**Examples of GOOD matching (high score):**
- User needs "glue that dries clear/fast" → Query mentions "rapid drying", "quick-drying formula", or "dries clear" (any of these = good)
- User needs "specific shades matching photos" → Query mentions "color matching", "exact shade", "precise color", or "color saturation" (any of these = good)
- User needs "bulk packs to avoid running out" → Query mentions "large quantity", "bulk pack", "multi-pack", or "high-volume" (any of these = good)

**Examples of BAD matching (low score):**
- Query mentions features the user doesn't care about (e.g., "heavy-duty durability" for a scrapbooker)
- Query is completely generic: "best product", "high quality", "good value" with NO specifics
- Query contradicts user's needs (e.g., "easy to use" when user prioritizes "visual perfection over usability")

**Scoring Criteria (1-10):**
- 1-2: Query completely misses user's unique preferences or contradicts their needs.
- 3-4: Query is generic (only "good quality", "durable", "versatile") OR focuses on needs user doesn't prioritize.
- 5-6: Query has some relevance but misses key distinctive traits or is too broad.
- 7-8: Query captures specific user needs well (e.g., drying time, color matching, bulk quantity).
- 9-10: Query perfectly matches multiple distinctive needs with precise details.

**IMPORTANT:**
- DO NOT penalize technical language - "chromatic saturation" and "color that doesn't fade" are BOTH valid if the user cares about color quality
- DO NOT reward generic terms - "premium quality", "excellent performance" deserve low scores unless accompanied by specific needs
- FOCUS on the USER'S DISTINCTIVE TRAITS from the persona: What do they care about that MOST users don't?

**Output Format:**
```json
{{
  "score": <1-10>,
  "justification": "<brief explanation>"
}}
```
"""
    response = call_llm_api_with_retry(prompt)
    if response:
        parsed = parse_llm_json_response(response)
        if parsed and 'score' in parsed:
            return {
                'score': int(parsed['score']),
                'justification': parsed.get('justification', '')
            }
    return {'score': 0, 'justification': 'Failed to parse response'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-queries-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/06_query")
    parser.add_argument("--persona-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/03_persona/results")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/10_evaluation_v2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load unique personas
    log_with_timestamp("Loading UNIQUE personas...")
    personas = {}
    for f in os.listdir(args.persona_dir):
        if f.startswith('persona_') and f.endswith('.json'):
            with open(os.path.join(args.persona_dir, f), 'r') as file:
                data = json.load(file)
                personas[data['user_id']] = data.get('persona', '')
    log_with_timestamp(f"Loaded {len(personas)} unique personas")

    # Load dual queries
    log_with_timestamp("Loading dual queries...")
    dual_query_files = sorted([
        os.path.join(args.dual_queries_dir, f)
        for f in os.listdir(args.dual_queries_dir)
        if f.startswith('dual_queries_') and f.endswith('.json')
    ])

    all_results = []

    for dual_file in dual_query_files:
        with open(dual_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            # Extract user_id from filename (dual_queries_USERID.json)
            user_id = os.path.basename(dual_file).replace('dual_queries_', '').replace('.json', '')
        else:
            user_id = data.get('user_id')
        user_persona = personas.get(user_id, '')

        if not user_persona:
            log_with_timestamp(f"Skipping {user_id} - no persona found")
            continue

        log_with_timestamp(f"\nEvaluating {user_id}...")

        results = []
        public_scores = []
        personalized_scores = []

        for i, item in enumerate(data):
            # Check for query existence using the actual keys
            if not (item.get('public_query') and item.get('personalized_query')):
                continue

            asin = item.get('asin')
            public_query = item.get('public_query')
            personalized_query = item.get('personalized_query')

            # Evaluate public query
            time.sleep(random.uniform(0.3, 0.6))
            public_result = evaluate_query_persona_match(public_query, user_persona, "Public Query")

            # Evaluate personalized query
            time.sleep(random.uniform(0.3, 0.6))
            personalized_result = evaluate_query_persona_match(personalized_query, user_persona, "Personalized Query")

            result = {
                'asin': asin,
                'category': item.get('category'),
                'public_query': public_query,
                'public_score': public_result['score'],
                'public_justification': public_result['justification'],
                'personalized_query': personalized_query,
                'personalized_score': personalized_result['score'],
                'personalized_justification': personalized_result['justification'],
                'score_diff': personalized_result['score'] - public_result['score']
            }
            results.append(result)

            public_scores.append(public_result['score'])
            personalized_scores.append(personalized_result['score'])

            log_with_timestamp(f"  [{i+1}] {asin}: Public={public_result['score']}, Personal={personalized_result['score']}, Diff={result['score_diff']}")

        # Save user results
        user_output = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'total_items': len(results),
            'avg_public_score': sum(public_scores) / len(public_scores) if public_scores else 0,
            'avg_personalized_score': sum(personalized_scores) / len(personalized_scores) if personalized_scores else 0,
            'results': results
        }

        output_file = os.path.join(args.output_dir, f"evaluation_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(user_output, f, indent=2, ensure_ascii=False)

        improvement = user_output['avg_personalized_score'] - user_output['avg_public_score']
        all_results.append({
            'user_id': user_id,
            'total': len(results),
            'avg_public': user_output['avg_public_score'],
            'avg_personalized': user_output['avg_personalized_score'],
            'improvement': improvement
        })

        log_with_timestamp(f"  Avg: Public={user_output['avg_public_score']:.2f}, Personalized={user_output['avg_personalized_score']:.2f}, Improve={improvement:+.2f}")

    # Summary
    log_with_timestamp("\n" + "="*70)
    log_with_timestamp("SUMMARY (Score: 1-10) - V2 NEEDS-BASED EVALUATION")
    log_with_timestamp("="*70)
    log_with_timestamp(f"{'User ID':<18} {'Items':>6} {'Public':>8} {'Personal':>8} {'Improve':>8}")
    log_with_timestamp("-"*70)

    total_public = 0
    total_personalized = 0
    total_items = 0

    for r in all_results:
        log_with_timestamp(f"{r['user_id']:<18} {r['total']:>6} {r['avg_public']:>8.2f} {r['avg_personalized']:>8.2f} {r['improvement']:>+8.2f}")
        total_public += r['avg_public'] * r['total']
        total_personalized += r['avg_personalized'] * r['total']
        total_items += r['total']

    log_with_timestamp("-"*70)
    overall_public = total_public / total_items if total_items > 0 else 0
    overall_personalized = total_personalized / total_items if total_items > 0 else 0
    overall_improve = overall_personalized - overall_public

    log_with_timestamp(f"{'OVERALL':<18} {total_items:>6} {overall_public:>8.2f} {overall_personalized:>8.2f} {overall_improve:>+8.2f}")
    log_with_timestamp("="*70)

    # Statistics
    improvements = [r['improvement'] for r in all_results]
    positive = sum(1 for i in improvements if i > 0)
    negative = sum(1 for i in improvements if i < 0)

    log_with_timestamp(f"\n改进情况:")
    log_with_timestamp(f"  个性化优于大众: {positive} 个用户 ({100*positive/len(improvements):.0f}%)")
    log_with_timestamp(f"  大众优于个性化: {negative} 个用户")

    # Save summary
    summary_output = {
        'timestamp': datetime.now().isoformat(),
        'score_range': '1-10',
        'evaluation_type': 'needs-based_v2',
        'total_items': total_items,
        'overall_public_score': overall_public,
        'overall_personalized_score': overall_personalized,
        'overall_improvement': overall_improve,
        'positive_count': positive,
        'negative_count': negative,
        'by_user': all_results
    }

    summary_file = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_output, f, indent=2, ensure_ascii=False)

    log_with_timestamp(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
