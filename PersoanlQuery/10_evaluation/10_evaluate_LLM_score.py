#!/usr/bin/env python3
"""
Stage 11: Evaluate Query-Persona Alignment with LLM

Reads dual queries from Stage 7 and persona files from Stage 4.
For each query pair, evaluates how well the query aligns with the
user's persona across 5 dimensions using LLM.

Input: 
  - result/personal_query/07_query/queries_{user_id}.json
  - result/personal_query/04_persona/persona_{Category}_{user_id}.json
  
Output: result/personal_query/11_evaluation/evaluation_{user_id}.json
"""

import json
import os
import sys
import re
import argparse
import random
import threading
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/fs04/ar57/wenyu/.claude/skills')
from llm_client import LLMClient


def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


# =============================================================================
# Dimension-Specific Evaluation Rules
# =============================================================================
# Each dimension has its own evaluation criteria based on its semantic meaning

DIMENSION_EVALUATION_RULES = {
    # Category 1: Product Attributes
    "Product_Category": {
        "description": "Product type/name",
        "rule": "Does the query specify or imply a product type that aligns with the user's product preference?",
        "positive_indicators": ["product name", "product type", "category of item", "specific craft supply"],
        "negative_indicators": ["vague", "generic", "no product mentioned"]
    },
    "Functionality": {
        "description": "Product features/capabilities",
        "rule": "Does the query mention product features or capabilities that the user wants?",
        "positive_indicators": ["can do", "able to", "has feature", "creates", "cuts", "embosses", "holds"],
        "negative_indicators": ["no feature mentioned", "generic"]
    },
    "Material_Composition": {
        "description": "Raw material/ingredient",
        "rule": "Does the query specify material preferences that match the user's preference?",
        "positive_indicators": ["metal", "plastic", "cotton", "wood", "paper", "fabric", "material type"],
        "negative_indicators": ["wrong material", "conflicting material"]
    },
    
    # Category 2: Quality Attributes
    "Quality_Craftsmanship": {
        "description": "Quality/workmanship",
        "rule": "Does the query express quality requirements that align with the user's quality expectations?",
        "positive_indicators": ["high quality", "well made", "sturdy", "durable", "quality", "professional"],
        "negative_indicators": ["cheap", "flimsy", "poor quality"]
    },
    "Performance": {
        "description": "Performance effectiveness",
        "rule": "Does the query mention performance expectations that match the user's needs?",
        "positive_indicators": ["works well", "cleans", "cuts cleanly", "performs", "effective", "reliable"],
        "negative_indicators": ["poor performance", "doesn't work"]
    },
    "Safety": {
        "description": "Safety requirements",
        "rule": "Does the query address safety concerns that match the user's safety preferences?",
        "positive_indicators": ["safe", "non-toxic", "secure", "child safe", "food safe"],
        "negative_indicators": ["unsafe", "dangerous", "hazardous"]
    },
    
    # Category 3: Appearance/Design
    "Appearance_Color": {
        "description": "Visual appearance",
        "rule": "Does the query specify color or appearance preferences that match the user's taste?",
        "positive_indicators": ["color", "silver", "gold", "metallic", "shiny", "cute", "beautiful", "pattern", "design"],
        "negative_indicators": ["wrong color", "ugly", "boring"]
    },
    "Size_Dimensions": {
        "description": "Size fit",
        "rule": "Does the query specify size requirements that match the user's dimensional needs?",
        "positive_indicators": ["size", "small", "large", "compact", "A2", "dimensions", "fits", "big", "tiny"],
        "negative_indicators": ["wrong size", "doesn't fit"]
    },
    "Style_Design": {
        "description": "Style preference",
        "rule": "Does the query express style preferences that match the user's design taste?",
        "positive_indicators": ["style", "modern", "vintage", "cute", "elegant", "intricate", "simple", "decorative"],
        "negative_indicators": ["wrong style", "don't want theme"]
    },
    
    # Category 4: User Experience
    "Comfort": {
        "description": "Comfort level",
        "rule": "Does the query mention comfort requirements that match the user's comfort preferences?",
        "positive_indicators": ["comfortable", "soft", "ergonomic", "easy to hold", "grip"],
        "negative_indicators": ["uncomfortable", "hard to hold"]
    },
    "Ease_of_Use": {
        "description": "Usability",
        "rule": "Does the query express ease-of-use requirements that match the user's preference for simplicity?",
        "positive_indicators": ["easy to use", "simple", "easy to assemble", "intuitive", "user friendly", "convenient"],
        "negative_indicators": ["complicated", "hard to use", "difficult"]
    },
    "Portability": {
        "description": "Portability",
        "rule": "Does the query mention portability needs that match the user's mobility requirements?",
        "positive_indicators": ["portable", "lightweight", "foldable", "travel", "compact", "on-the-go"],
        "negative_indicators": ["bulky", "heavy", "not portable"]
    },
    
    # Category 5: Usage Scenarios
    "Target_User": {
        "description": "Intended user",
        "rule": "Does the query indicate it is for a specific user type that matches who the persona is?",
        "positive_indicators": ["for beginners", "for kids", "for professionals", "for me", "my child", "teacher"],
        "negative_indicators": ["wrong user type"]
    },
    "Usage_Scenario": {
        "description": "Where/how to use",
        "rule": "Does the query mention usage context that matches the user's actual use case?",
        "positive_indicators": ["for home", "for classroom", "for scrapbooking", "for card making", "at work", "outdoor"],
        "negative_indicators": ["wrong context", "doesn't match"]
    },
    "Special_Purpose": {
        "description": "Special use case",
        "rule": "Does the query mention a special purpose that aligns with the user's specific needs?",
        "positive_indicators": ["for gift", "for party", "for charity", "for holiday", "for wedding", "specific purpose"],
        "negative_indicators": ["wrong purpose"]
    },
    
    # Category 6: Price/Value
    "Price": {
        "description": "Price related",
        "rule": "Does the query express price expectations that match the user's budget?",
        "positive_indicators": ["affordable", "cheap", "expensive", "budget", "price", "under $", "cheap"],
        "negative_indicators": []
    },
    "Value": {
        "description": "Value for money",
        "rule": "Does the query express value expectations that match the user's value orientation?",
        "positive_indicators": ["worth", "value", "deal", "bundle", "set", "pack", "quantity"],
        "negative_indicators": ["poor value", "not worth"]
    },
    "Packaging_Quantity": {
        "description": "Packaging specs",
        "rule": "Does the query specify quantity or packaging preferences that match the user's buying habits?",
        "positive_indicators": ["pack", "set", "bundle", "multiple", "quantity", "bulk", "single", "bottles", "pieces"],
        "negative_indicators": []
    },
    
    # Category 7: Special Requirements
    "Compatibility": {
        "description": "Device/system compatibility",
        "rule": "Does the query mention compatibility requirements that match the user's existing equipment?",
        "positive_indicators": ["works with", "compatible with", "fits", "for Cuttlebug", "for Sizzix", "adapter"],
        "negative_indicators": ["incompatible", "doesn't fit"]
    },
    "Special_User_Needs": {
        "description": "Special user requirements",
        "rule": "Does the query address special needs that match the user's specific requirements?",
        "positive_indicators": ["sensitive", "allergy", "special need", "requirement"],
        "negative_indicators": []
    },
    "Brand_Preference": {
        "description": "Brand preference",
        "rule": "Does the query mention brand preferences that match the user's brand loyalty?",
        "positive_indicators": ["Fiskars", "Martha Stewart", "brand name", "name brand", "specific brand"],
        "negative_indicators": ["wrong brand", "avoid brand"]
    }
}


def normalize_category(category: str) -> str:
    """Normalize category name to match persona file naming."""
    return category.replace(" ", "_").replace(",", "").replace("&", "and")


def load_persona(persona_dir: str, user_id: str, category: str) -> Optional[Dict]:
    """
    Load persona file for a specific user and category.
    
    Args:
        persona_dir: Directory containing persona files
        user_id: User ID
        category: Product category
        
    Returns:
        Dict with dimension_personas or None if not found
    """
    normalized_cat = normalize_category(category)
    persona_file = os.path.join(persona_dir, f"persona_{normalized_cat}_{user_id}.json")
    
    if not os.path.exists(persona_file):
        log_with_timestamp(f"    Warning: Persona file not found: {persona_file}")
        return None
    
    with open(persona_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_mass_persona(mass_persona_dir: str, category: str) -> Optional[Dict]:
    """
    Load mass market persona file for a category.
    
    Args:
        mass_persona_dir: Directory containing mass market persona files
        category: Product category
        
    Returns:
        Dict with dimension_personas or None if not found
    """
    normalized_cat = normalize_category(category)
    persona_file = os.path.join(mass_persona_dir, f"persona_{normalized_cat}_mass_market.json")
    
    if not os.path.exists(persona_file):
        log_with_timestamp(f"    Warning: Mass persona file not found: {persona_file}")
        return None
    
    with open(persona_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_relevant_personas(persona_data: Dict, dimensions: List[str]) -> Dict[str, str]:
    """
    Extract persona descriptions for the specified dimensions.
    
    Args:
        persona_data: Full persona data with dimension_personas
        dimensions: List of dimension names to extract
        
    Returns:
        Dict mapping dimension name to persona description
    """
    dimension_personas = persona_data.get('dimension_personas', {})
    result = {}
    
    for dim in dimensions:
        if dim in dimension_personas:
            result[dim] = dimension_personas[dim]
        else:
            result[dim] = "No specific persona information available."
    
    return result


def create_evaluation_prompt(query: str, personas: Dict[str, str], 
                             dimensions: List[str], query_type: str) -> str:
    """
    Create LLM prompt for evaluating query-persona alignment.
    Uses dimension-specific evaluation rules for accurate assessment.
    
    Args:
        query: The query text to evaluate
        personas: Dict mapping dimension to persona description
        dimensions: List of dimension names (from query generation)
        query_type: 'target_user', 'mass_market', 'cross_qu_pg', 'cross_qg_pg'
        
    Returns:
        Prompt string for LLM
    """
    dimension_context = []
    for dim in dimensions:
        persona_desc = personas.get(dim, "No persona information")
        rule_info = DIMENSION_EVALUATION_RULES.get(dim, {})
        rule_text = rule_info.get("rule", "Does the query reflect the user's preference?")
        
        dimension_context.append(
            f"**{dim}** ({rule_info.get('description', 'N/A')}):\n"
            f"  Persona: {persona_desc}\n"
            f"  Evaluation: {rule_text}"
        )
    
    persona_text = "\n\n".join(dimension_context)
    
    rules_text = "\n".join([
        f'  "{dim}_alignment": {{"passed": true/false, "reason": "..."}}'
        for dim in dimensions
    ])
    
    prompt = f"""You MUST output ONLY valid JSON. No explanations. No thinking. No markdown. Just JSON.

For each dimension used to generate the query, evaluate if the query aligns with the persona's preference.

{persona_text}

Query: {query}

For EACH dimension above, determine if the query reflects that specific preference from the persona.
Output JSON with one key per dimension:
{{
{rules_text},
  "total_score": <number of passed dimensions>
}}

Output JSON:"""

    return prompt


def parse_llm_response(response: str, dimensions: List[str] = None) -> Dict:
    """
    Parse LLM response, handling thinking blocks and extracting JSON.
    
    Args:
        response: LLM response text
        dimensions: List of dimension names used in the query
        
    Returns:
        Dict with evaluation results and parse status
    """
    if not response:
        return {"total_score": 0, "error": "Empty response", "parse_error": True}
    
    if dimensions is None:
        dimensions = []
        return {"total_score": 0, "error": "Empty response", "parse_error": True}
    
    # Step 1: Remove thinking blocks (MiniMax/GLM style)
    clean_response = re.sub(r'<think>[\s\S]*?</think>', '', response)
    clean_response = re.sub(r'<thinking>[\s\S]*?</thinking>', '', clean_response)
    clean_response = re.sub(r'<tool_code>[\s\S]*?</tool_code>', '', clean_response)
    clean_response = clean_response.strip()
    
    # Step 2: Remove markdown code blocks
    clean_response = re.sub(r'```json', '', clean_response)
    clean_response = re.sub(r'```', '', clean_response)
    clean_response = clean_response.strip()
    
    # Step 3: Try to extract valid JSON
    # Look for JSON object with proper structure
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', clean_response)
    if json_match:
        try:
            result = json.loads(json_match.group())
            # Calculate total score if not provided
            if 'total_score' not in result:
                score = 0
                for key in result:
                    if isinstance(result[key], dict) and 'passed' in result[key]:
                        if result[key]['passed']:
                            score += 1
                result['total_score'] = score
            
            result['parse_error'] = False
            return result
        except (json.JSONDecodeError, Exception) as e:
            # JSON extracted but invalid, continue to fallback
            pass
    
    # Step 4: Fallback - extract structured YES/NO for each dimension
    score = 0
    rules = {}
    
    # Build dynamic patterns based on actual dimensions used in query
    if not dimensions:
        dimensions = []
    
    rule_patterns = []
    for dim in dimensions:
        dim_key = f"{dim}_alignment"
        rule_patterns.append((dim_key, [
            rf'{re.escape(dim)}[^a-zA-Z]*(?:YES|yes|true|True|PASS|Pass|符合|通过)',
            rf'{re.escape(dim).replace("_", " ")}[^a-zA-Z]*(?:YES|yes|true|True|PASS|Pass|符合|通过)',
            r'alignment[^a-zA-Z]*(?:YES|yes|true|True|PASS|Pass|符合|通过)'
        ]))
    
    for rule_key, patterns in rule_patterns:
        matched = False
        for pattern in patterns:
            if re.search(pattern, clean_response, re.IGNORECASE):
                score += 1
                matched = True
                break
        rules[rule_key] = {"passed": matched, "reason": "Extracted from fallback"}
    
    # Step 5: If no rules matched, try score extraction
    if score == 0:
        # Look for numeric scores like "total_score": 3 or "score": 3
        score_match = re.search(r'(?:total_)?score["\s:]+(\d+)', clean_response, re.IGNORECASE)
        if score_match:
            score = min(int(score_match.group(1)), 5)
        else:
            # Last resort: count positive indicators
            positive_count = len(re.findall(
                r'\b(YES|yes|true|True|PASS|Pass|符合|通过|PASSED|passed)\b', 
                clean_response
            ))
            negative_count = len(re.findall(
                r'\b(NO|no|false|False|FAIL|Fail|不符合|未通过|FAILED|failed)\b', 
                clean_response
            ))
            score = max(0, min(positive_count - negative_count, 5))
    
    return {
        "total_score": score,
        **rules,
        "raw_response": response[:500],
        "parse_error": True
    }


def evaluate_query(query: str, personas: Dict[str, str], 
                   dimensions: List[str], query_type: str,
                   max_retries: int = 5) -> Dict:
    """
    Evaluate a single query against persona using LLM.
    
    Args:
        query: Query text to evaluate
        personas: Dict mapping dimension to persona description
        dimensions: List of dimension names
        query_type: 'target_user' or 'mass_market'
        max_retries: Number of retry attempts
        
    Returns:
        Dict with evaluation results
    """
    if not query:
        return {
            "total_score": 0,
            "error": "No query provided"
        }
    
    client = LLMClient()
    prompt = create_evaluation_prompt(query, personas, dimensions, query_type)
    
    for attempt in range(max_retries + 1):
        try:
            response = client.call(prompt, max_tokens=2048, temperature=0.0)
            if response:
                result = parse_llm_response(response, dimensions)
                result['query_type'] = query_type
                result['attempts'] = attempt + 1
                return result
        except Exception as e:
            if attempt == max_retries:
                log_with_timestamp(f"      LLM error after {max_retries + 1} attempts: {e}")
                return {
                    "total_score": 0,
                    "error": str(e),
                    "query_type": query_type
                }
            continue
    
    return {
        "total_score": 0,
        "error": "All LLM attempts failed",
        "query_type": query_type
    }


def process_query_pair(args: Tuple) -> Tuple[int, Dict]:
    """
    Process a single query pair (target + mass market).
    
    Args:
        args: Tuple of (index, query_result, persona_dir, mass_persona_dir, user_id)
        
    Returns:
        Tuple of (index, evaluation_result)
    """
    index, query_result, persona_dir, mass_persona_dir, user_id = args
    
    asin = query_result.get('asin', 'Unknown')
    category = query_result.get('category', 'Unknown')
    dimensions = query_result.get('shared_dimensions', [])
    
    log_with_timestamp(f"\n[{index+1}] Evaluating ASIN: {asin} ({category})")
    log_with_timestamp(f"    Dimensions: {dimensions}")
    
    # Load persona for this category
    persona_data = load_persona(persona_dir, user_id, category)
    
    if not persona_data:
        return index, {
            "asin": asin,
            "category": category,
            "error": "Persona not found",
            "target_evaluation": {"total_score": 0},
            "mass_market_evaluation": {"total_score": 0}
        }
    
    # Get relevant persona descriptions
    personas = get_relevant_personas(persona_data, dimensions)
    
    # Get query texts
    target_query = query_result.get('target_user_query', {}).get('query', '')
    mass_query = query_result.get('mass_market_query', {}).get('query', '')
    
    result = {
        "asin": asin,
        "category": category,
        "shared_dimensions": dimensions,
        "dimension_personas": personas
    }
    
    # Evaluate target user query
    if target_query:
        log_with_timestamp(f"    Evaluating target user query...")
        target_eval = evaluate_query(target_query, personas, dimensions, 'target_user')
        result['target_user_query'] = target_query
        result['target_evaluation'] = target_eval
        log_with_timestamp(f"    Target score: {target_eval.get('total_score', 0)}/5")
    else:
        result['target_user_query'] = ""
        result['target_evaluation'] = {"total_score": 0, "error": "No target query"}
    
    # Evaluate mass market query
    if mass_query:
        log_with_timestamp(f"    Evaluating mass market query...")
        mass_eval = evaluate_query(mass_query, personas, dimensions, 'mass_market')
        result['mass_market_query'] = mass_query
        result['mass_market_evaluation'] = mass_eval
        log_with_timestamp(f"    Mass market score: {mass_eval.get('total_score', 0)}/5")
    else:
        result['mass_market_query'] = ""
        result['mass_market_evaluation'] = {"total_score": 0, "error": "No mass market query"}
    
    # Load mass market persona and evaluate cross-matrix
    mass_persona_data = load_mass_persona(mass_persona_dir, category)
    mass_personas = get_relevant_personas(mass_persona_data, dimensions) if mass_persona_data else {}
    
    result['cross_evaluation'] = {}
    
    if mass_personas and target_query and mass_query:
        log_with_timestamp(f"    Evaluating S(Q_u, P_g)...")
        qu_pg_eval = evaluate_query(target_query, mass_personas, dimensions, 'cross_qu_pg')
        result['cross_evaluation']['S_qu_pg'] = qu_pg_eval.get('total_score', 0)
        
        log_with_timestamp(f"    Evaluating S(Q_g, P_g)...")
        qg_pg_eval = evaluate_query(mass_query, mass_personas, dimensions, 'cross_qg_pg')
        result['cross_evaluation']['S_qg_pg'] = qg_pg_eval.get('total_score', 0)
    
    # Determine winner
    target_score = result['target_evaluation'].get('total_score', 0)
    mass_score = result['mass_market_evaluation'].get('total_score', 0)
    
    # Calculate Personalization Gain
    # Gain = Score(Q_u, P_u) - Score(Q_g, P_u)
    personalization_gain = target_score - mass_score
    result['personalization_gain'] = personalization_gain
    
    # Calculate DiD Relative Gain
    # Gain_relative = [S(Q_u, P_u) - S(Q_g, P_u)] - [S(Q_u, P_g) - S(Q_g, P_g)]
    s_qu_pu = target_score
    s_qg_pu = mass_score
    s_qu_pg = result['cross_evaluation'].get('S_qu_pg', 0)
    s_qg_pg = result['cross_evaluation'].get('S_qg_pg', 0)
    
    direct_gain = s_qu_pu - s_qg_pu
    cross_gain = s_qu_pg - s_qg_pg
    relative_gain = direct_gain - cross_gain
    result['relative_gain'] = relative_gain
    result['direct_gain'] = direct_gain
    
    if target_score > mass_score:
        result['winner'] = 'target_user'
    elif mass_score > target_score:
        result['winner'] = 'mass_market'
    else:
        result['winner'] = 'tie'
    
    return index, result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate query-persona alignment using LLM"
    )
    parser.add_argument(
        "--input-file",
        default="/fs04/ar57/wenyu/result/personal_query/07_query/queries_A13OFOB1394G31.json",
        help="Input dual queries JSON file"
    )
    parser.add_argument(
        "--persona-dir",
        default="/fs04/ar57/wenyu/result/personal_query/04_persona",
        help="Directory containing persona files"
    )
    parser.add_argument(
        "--mass-persona-dir",
        default="/fs04/ar57/wenyu/result/personal_query/04_persona_mass",
        help="Directory containing mass market persona files"
    )
    parser.add_argument(
        "--output-dir",
        default="/fs04/ar57/wenyu/result/personal_query/11_evaluation",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=50,
        help="Number of concurrent workers (default: 50)"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input file
    log_with_timestamp(f"Loading input file: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    user_id = data.get('user_id', 'Unknown')
    query_results = data.get('results', [])
    
    log_with_timestamp(f"User ID: {user_id}")
    log_with_timestamp(f"Total query pairs to evaluate: {len(query_results)}")
    log_with_timestamp(f"Using {args.workers} concurrent workers")
    
    # Prepare tasks
    tasks = [(i, qr, args.persona_dir, args.mass_persona_dir, user_id) for i, qr in enumerate(query_results)]
    
    # Process concurrently
    results = [None] * len(query_results)
    target_wins = 0
    mass_wins = 0
    ties = 0
    total_target_score = 0
    total_mass_score = 0
    total_personalization_gain = 0
    total_relative_gain = 0
    evaluated_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_index = {
            executor.submit(process_query_pair, task): task[0]
            for task in tasks
        }
        
        for future in as_completed(future_to_index):
            try:
                index, result = future.result()
                results[index] = result
                evaluated_count += 1
                
                # Track statistics
                winner = result.get('winner', 'tie')
                if winner == 'target_user':
                    target_wins += 1
                elif winner == 'mass_market':
                    mass_wins += 1
                else:
                    ties += 1
                
                target_score = result.get('target_evaluation', {}).get('total_score', 0)
                mass_score = result.get('mass_market_evaluation', {}).get('total_score', 0)
                personalization_gain = result.get('personalization_gain', 0)
                relative_gain = result.get('relative_gain', 0)
                total_target_score += target_score
                total_mass_score += mass_score
                total_personalization_gain += personalization_gain
                total_relative_gain += relative_gain
                
            except Exception as e:
                index = future_to_index[future]
                log_with_timestamp(f"  Error processing item {index}: {e}")
                results[index] = {
                    "asin": "Error",
                    "error": str(e),
                    "target_evaluation": {"total_score": 0},
                    "mass_market_evaluation": {"total_score": 0}
                }
    
    # Calculate statistics
    avg_target_score = total_target_score / evaluated_count if evaluated_count > 0 else 0
    avg_mass_score = total_mass_score / evaluated_count if evaluated_count > 0 else 0
    avg_personalization_gain = total_personalization_gain / evaluated_count if evaluated_count > 0 else 0
    avg_relative_gain = total_relative_gain / evaluated_count if evaluated_count > 0 else 0
    
    # Build output data
    output_data = {
        "user_id": user_id,
        "timestamp": datetime.now().isoformat(),
        "input_file": args.input_file,
        "persona_dir": args.persona_dir,
        "mass_persona_dir": args.mass_persona_dir,
        "total_evaluated": evaluated_count,
        "statistics": {
            "target_user_wins": target_wins,
            "mass_market_wins": mass_wins,
            "ties": ties,
            "total_target_score": total_target_score,
            "total_mass_score": total_mass_score,
            "total_personalization_gain": total_personalization_gain,
            "total_relative_gain": total_relative_gain,
            "average_target_score": round(avg_target_score, 2),
            "average_mass_score": round(avg_mass_score, 2),
            "average_personalization_gain": round(avg_personalization_gain, 2),
            "average_relative_gain": round(avg_relative_gain, 2),
            "target_win_rate": round(target_wins / evaluated_count * 100, 2) if evaluated_count > 0 else 0,
            "mass_win_rate": round(mass_wins / evaluated_count * 100, 2) if evaluated_count > 0 else 0
        },
        "evaluation_rules": {
            "rule_1": "Specific Preference Alignment - Does the query reflect the user's specific preferences?",
            "rule_2": "Vocabulary Consistency - Does the query use terminology consistent with the persona?",
            "rule_3": "Pain Point Addressing - Does the query address the user's stated needs/concerns?",
            "rule_4": "Usage Scenario Match - Does the query align with how the persona uses products?",
            "rule_5": "Overall Persona Fit - Does the query feel like it could come from this user?"
        },
        "results": results
    }
    
    # Save output
    output_file = os.path.join(args.output_dir, f"evaluation_{user_id}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"\n{'='*60}")
    log_with_timestamp(f"Evaluation Summary:")
    log_with_timestamp(f"  Total evaluated: {evaluated_count}")
    log_with_timestamp(f"  Target User Wins: {target_wins} ({output_data['statistics']['target_win_rate']}%)")
    log_with_timestamp(f"  Mass Market Wins: {mass_wins} ({output_data['statistics']['mass_win_rate']}%)")
    log_with_timestamp(f"  Ties: {ties}")
    log_with_timestamp(f"  Average Target Score: {avg_target_score:.2f}/5")
    log_with_timestamp(f"  Average Mass Market Score: {avg_mass_score:.2f}/5")
    log_with_timestamp(f"  Average Personalization Gain: {avg_personalization_gain:.2f}")
    log_with_timestamp(f"  Average Relative Gain (DiD): {avg_relative_gain:.2f}")
    log_with_timestamp(f"  Output saved to: {output_file}")
    log_with_timestamp(f"{'='*60}")


if __name__ == "__main__":
    main()
