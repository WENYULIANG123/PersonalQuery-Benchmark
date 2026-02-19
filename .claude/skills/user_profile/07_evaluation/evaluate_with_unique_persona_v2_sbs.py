import json
import os
import argparse
import time
import random
import re
from datetime import datetime

# Import LLM client
import sys
sys.path.append('/home/wlia0047/ar57/wenyu/.claude/skills')
from llm_client import LLMClient

# Rate limiter global
last_call_time = 0

def call_llm_api_with_retry(prompt, max_retries=3):
    global last_call_time
    client = LLMClient()
    
    for attempt in range(max_retries):
        try:
            # Simple rate limiting: ensure at least 0.5s between calls
            current_time = time.time()
            if current_time - last_call_time < 0.5:
                time.sleep(0.5 - (current_time - last_call_time))
            
            response = client.call(prompt)
            last_call_time = time.time()
            return response
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error calling LLM API after {max_retries} attempts: {e}")
                return None
            time.sleep(2 ** attempt)  # Exponential backoff

def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def parse_llm_json_response(llm_response):
    try:
        json_content = llm_response
        if "```json" in llm_response:
            json_content = llm_response.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_response:
            json_content = llm_response.split("```")[1].split("```")[0].strip()
        else:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_content = json_match.group(0)
        return json.loads(json_content)
    except:
        return None

def evaluate_sbs(persona, query_a, query_b, user_id=None, asin=None):
    """
    Side-by-Side evaluation of two queries against a user persona.
    Returns: 'A', 'B', or 'Tie'
    """
    prompt = f"""You are an expert search relevance evaluator. Your task is to compare two search queries and determine which one better reflects a specific user's UNIQUE persona.

**User Persona:**
{persona}

**Query A:** "{query_a}"
**Query B:** "{query_b}"

**Evaluation Criteria:**
1. **Specificity:** Does the query capture unique preferences (e.g., specific brands, materials, techniques) mentioned in the persona?
2. **Intent Alignment:** Does the query reflect the user's specific shopping goals or project needs?
3. **Avoidance of Generics:** Penalize queries that rely solely on generic terms like 'good quality' or 'best' if the persona allows for more specific details.

**Task:**
Which query is BETTER tailored to this specific user?
- **Select 'A'** if Query A is clearly more specific and aligned with the unique persona.
- **Select 'B'** if Query B is clearly more specific and aligned with the unique persona.
- **Select 'Tie'** if both are equally good or equally bad/generic.

**Output Format:**
```json
{{
  "winner": "A" or "B" or "Tie",
  "reason": "Brief explanation of why the winner is better, specifically referencing the persona."
}}
```
"""
    response = call_llm_api_with_retry(prompt)
    if response:
        parsed = parse_llm_json_response(response)
        if parsed and 'winner' in parsed:
            return {
                'winner': parsed['winner'],
                'reason': parsed.get('reason', '')
            }
    return {'winner': 'Error', 'reason': 'Failed to parse response'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-queries-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/new_dual_queries")
    parser.add_argument("--persona-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/new_persona_results")
    parser.add_argument("--output-dir",
                        default="/home/wlia0047/ar57/wenyu/result/user_profile/new_evaluation_sbs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load unique personas
    log_with_timestamp("Loading UNIQUE personas...")
    personas = {}
    if os.path.exists(args.persona_dir):
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
    
    # Statistics
    total_comparisons = 0
    personalized_wins = 0
    public_wins = 0
    ties = 0

    for dual_file in dual_query_files:
        with open(dual_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract user_id
        if isinstance(data, list):
             # Try to get user_id from filename as fallback if not in list items (it's not usually)
            filename = os.path.basename(dual_file)
            user_id = filename.replace('dual_queries_', '').replace('.json', '')
        else:
            # Should be a list based on generation script, but handle dict if wrapper
             user_id = data.get('user_id', 'unknown')
             data = data.get('queries', []) # Handle potential wrapper

        user_persona = personas.get(user_id, '')

        if not user_persona:
            log_with_timestamp(f"Skipping {user_id} - no persona found")
            continue

        log_with_timestamp(f"\nEvaluating {user_id} (Side-by-Side)...")

        user_results = []
        user_p_wins = 0
        user_comparisons = 0

        for i, item in enumerate(data):
            if not (item.get('public_query') and item.get('personalized_query')):
                continue

            asin = item.get('asin')
            public_query = item.get('public_query')
            personalized_query = item.get('personalized_query')
            
            # Randomize order to avoid position bias
            is_personal_A = random.choice([True, False])
            
            if is_personal_A:
                query_a = personalized_query
                query_b = public_query
                personal_label = "A"
            else:
                query_a = public_query
                query_b = personalized_query
                personal_label = "B"

            # Evaluate Side-by-Side
            time.sleep(random.uniform(0.3, 0.6))
            sbs_result = evaluate_sbs(user_persona, query_a, query_b, user_id, asin)
            winner = sbs_result['winner']
            
            # Determine actual winner type
            winner_type = "Tie"
            if winner == personal_label:
                winner_type = "Personalized"
                personalized_wins += 1
                user_p_wins += 1
            elif winner == "Tie":
                winner_type = "Tie"
                ties += 1
            elif winner != "Error": # public wins
                winner_type = "Public"
                public_wins += 1
            
            total_comparisons += 1
            user_comparisons += 1
            
            log_with_timestamp(f"  [{i+1}] {asin}: Winner={winner_type} ({winner})")
            
            user_results.append({
                'asin': asin,
                'public_query': public_query,
                'personalized_query': personalized_query,
                'winner': winner_type,
                'reason': sbs_result['reason']
            })

            # Real-time stats logging per user
            if user_comparisons == 10:
                log_with_timestamp(f"  > Win Rate for {user_id}: {user_p_wins/user_comparisons*100:.1f}%")

        # Save user results
        output_file = os.path.join(args.output_dir, f"sbs_evaluation_{user_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'user_id': user_id, 
                'win_rate': user_p_wins / user_comparisons if user_comparisons > 0 else 0,
                'results': user_results
            }, f, indent=2)

    # Summary
    log_with_timestamp("\n" + "="*70)
    log_with_timestamp("SIDE-BY-SIDE EVALUATION SUMMARY")
    log_with_timestamp("="*70)
    log_with_timestamp(f"Total Comparisons: {total_comparisons}")
    log_with_timestamp(f"Personalized Wins: {personalized_wins} ({personalized_wins/total_comparisons*100:.1f}%)")
    log_with_timestamp(f"Public Wins:       {public_wins} ({public_wins/total_comparisons*100:.1f}%)")
    log_with_timestamp(f"Ties:              {ties} ({ties/total_comparisons*100:.1f}%)")
    log_with_timestamp("="*70)

    # Save summary
    summary_data = {
        'total_comparisons': total_comparisons,
        'personalized_wins': personalized_wins,
        'public_wins': public_wins,
        'ties': ties,
        'win_rate': personalized_wins/total_comparisons if total_comparisons > 0 else 0
    }
    with open(os.path.join(args.output_dir, "sbs_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)

if __name__ == "__main__":
    main()
