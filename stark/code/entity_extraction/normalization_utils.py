import json
import asyncio
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm as async_tqdm
from datetime import datetime

# Shared Schema
SCHEMA = {
    "Color": ["Blue", "Green", "Red", "Yellow", "Purple", "Brown", "Gray", "White", "Black", "Pink", "Metallic"],
    "Material": ["Textile", "Metal", "Plastic", "Wood", "Paper", "Mineral", "Glass", "Medium"],
    "Usage": ["Art", "Craft", "Sew", "Write", "Card", "Technical", "Office", "Storage", "Activity"],
    "Dimensions": ["Small", "Medium", "Large"],
    "Quantity": ["Single", "Bulk"],
    "Safety/Certification": ["Safe", "Professional", "Toxic"],
    "Design": ["Shape", "Pattern", "Style", "Format"],
    "Selling Point": ["Quality", "Feature", "Portable", "Origin"],
    "Technique": ["Blending", "Drying", "Washing", "Detailing", "Storage"],
    "Accessories": ["Tool", "Storage", "Refill"]
}

def normalize_entity_key(key: str) -> str:
    """Normalize entity type keys to match SCHEMA."""
    key_lower = key.lower().strip()
    if key_lower in ["color/finish", "colour/finish", "colors", "colours"]: return "Color"
    elif key_lower in ["selling_point", "selling points"]: return "Selling Point"
    elif key_lower in ["size", "dimension"]: return "Dimensions"
    elif key_lower in ["technique", "techniques", "method"]: return "Technique"
    elif key_lower in ["activity", "activities", "usage"]: return "Usage"
    elif key_lower in ["materials"]: return "Material"
    elif key_lower in ["accessory", "accessories"]: return "Accessories"
    return key.strip().title() # Default to title case

def build_generic_classification_prompt(entities: Dict[str, Any], schema: Dict = SCHEMA) -> str:
    """Build the standard prompt for entity classification."""
    return f"""
You are an expert data normalization assistant for e-commerce products.
Your task is to classify raw product attribute values into standardized categories.

**Available Schemas:**
{json.dumps(schema, indent=2)}

**Entities to Classify:**
{json.dumps(entities, indent=2)}

**CRITICAL INSTRUCTIONS:**
1. For each entity category and its values, you MUST choose the BEST matching class from the corresponding schema.
2. **DO NOT use "Other" as a classification.** You must select one of the explicit classes listed in the schema.
3. If a value doesn't perfectly match, choose the CLOSEST or MOST RELATED class from the available options.
4. Use your semantic understanding to find the best fit - consider synonyms, related concepts, and context.
5. **Contextual Hint**: For artistic products, if a usage involves making something, prefer "Craft". If it involves artistic expression, prefer "Art".
6. **Specific matches > Generic matches**: If a value describes a specific Design (e.g. "Cable Chain", "Round Cut"), classify as "Design" -> "Style/Shape", NOT "Selling Point" -> "Feature".
7. **Material vs Color**: "Gold" can be a Material or Color. Use context. if "14k Gold", it's Material.
8. **IMPORTANT**: Only return null (None) if an input value is purely junk, totally irrelevant to the category, or impossible to map to the broad classes provided.
7. Return a JSON object with the same structure as input, but with normalized values.

**Output Format:**
{{
  "Category1": {{"original_value1": "NormalizedClass1", "original_value2": "NormalizedClass2"}},
  "Category2": {{"original_value3": "NormalizedClass3"}}
}}

Return ONLY the JSON object, no explanations.
"""

def execute_llm_classification(llm_model, prompt: str, product_id: str, context: str) -> tuple[dict, dict]:
    """Execute LLM call using model.py infrastructure."""
    from model import call_llm_with_retry, _call_llm_with_openai_client, ApiProvider, get_api_provider, get_siliconflow_config, get_model_name, _resolve_thinking_budget, _save_api_response, get_current_api_info
    
    # Try using OpenAI client directly for better control if using SiliconFlow
    try:
        provider_info = get_api_provider()
        if provider_info['provider'] == ApiProvider.SILICONFLOW:
            # Direct client usage
            raw_response_dict, error = _call_llm_with_openai_client(
                prompt=prompt,
                model_name=get_model_name(),
                base_url=get_siliconflow_config()['base_url'],
                api_key=provider_info['api_key'],
                temperature=getattr(llm_model, 'temperature', 0.1),
                max_tokens=getattr(llm_model, 'max_tokens', 1000),
                thinking_budget=_resolve_thinking_budget(llm_model)
            )
            
            # Save response
            _save_api_response(
                context=context,
                prompt=prompt,
                response_dict=raw_response_dict,
                api_info=get_current_api_info(),
                success=not error,
                error=error,
                meta={"id": product_id}
            )
            
            if error:
                 raise Exception(f"API Error: {error}")
            
            response_str = raw_response_dict.get('content', '')
        else:
            # Fallback to LangChain via call_llm_with_retry
            response_str, success = call_llm_with_retry(llm_model, prompt, context=context, meta={"id": product_id})
            raw_response_dict = {"content": response_str}
            if not success:
                 raise Exception("LLM Call Failed")

        # Parse JSON
        clean_str = response_str.strip()
        if "```json" in clean_str:
            clean_str = clean_str.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_str:
            clean_str = clean_str.split("```")[1].split("```")[0].strip()
            
        return json.loads(clean_str), raw_response_dict

    except Exception as e:
        raise e

async def process_item_async(executor, process_func, *args):
    """Run a synchronous process function in an executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, process_func, *args)
