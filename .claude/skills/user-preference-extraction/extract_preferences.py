#!/usr/bin/env python3
"""
User Preference Extraction Module
Core logic for extracting structured user preferences from unstructured reviews.
"""

import json
import re
from typing import Dict, List, Optional, Union, Tuple
import sys

class ExtractionException(Exception):
    """Base exception for extraction errors"""
    pass

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

def clean_html_content(text) -> str:
    """Remove HTML tags and clean up content for entity extraction."""
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove JavaScript content
    text = re.sub(r'javascript:[^\'"\\s]*', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove common Amazon UI text patterns
    text = re.sub(r'Save \d+% on.*?(?=when|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Enter code [A-Z0-9]+ at checkout', '', text, flags=re.IGNORECASE)
    text = re.sub(r'restrictions apply', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Here\'s how', '', text, flags=re.IGNORECASE)

    # Clean up extra spaces again
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def _normalize_sentiment(val: Optional[str]) -> str:
    if not val:
        return "neutral"
    v = str(val).strip().lower()
    if v in {"positive", "pos", "+"}:
        return "positive"
    if v in {"negative", "neg", "-"}:
        return "negative"
    if v in {"neutral", "neu", "unknown", "mixed"}:
        return "neutral"
    return "neutral"

def _coerce_entity_item(item) -> Optional[Dict[str, str]]:
    """
    Normalize entity item to: {"entity": <str>, "sentiment": <positive|negative|neutral>, "original_text": <str>, "improvement_wish": <str|None>}
    """
    if isinstance(item, str):
        entity_text = item.strip()
        if not entity_text:
            return None
        return {"entity": entity_text, "sentiment": "neutral", "original_text": entity_text}

    if isinstance(item, dict):
        entity_text = (
            item.get("value") or item.get("entity")
            or item.get("text")
            or item.get("name")
            or ""
        )
        entity_text = str(entity_text).strip()
        if not entity_text:
            return None
        sentiment = _normalize_sentiment(item.get("sentiment") or item.get("polarity"))
        
        # Extract original text if provided, otherwise fallback to entity text
        original_text = item.get("original_text") or item.get("original_cvalue") or item.get("mention") or entity_text
        original_text = str(original_text).strip()
        
        # Extract improvement wish
        improvement_wish = item.get("improvement_wish")
        if improvement_wish:
            improvement_wish = str(improvement_wish).strip()
        
        result = {"entity": entity_text, "sentiment": sentiment, "original_text": original_text}
        if improvement_wish:
            result["improvement_wish"] = improvement_wish
            
        return result

    return None

def normalize_user_preference_entities_with_sentiment(entities: Union[List, Dict]) -> Union[List, Dict]:
    """
    Normalize user preference entities, maintaining structure (list or categorized dict).
    """
    if not entities:
        return entities

    if isinstance(entities, list):
        normalized = []
        for item in entities:
            coerced = _coerce_entity_item(item)
            if coerced:
                normalized.append(coerced)
        return normalized

    if isinstance(entities, dict):
        normalized = {}
        for category, items in entities.items():
            if isinstance(items, list):
                category_normalized = []
                for item in items:
                    coerced = _coerce_entity_item(item)
                    if coerced:
                        category_normalized.append(coerced)
                if category_normalized:
                    normalized[category] = category_normalized
            else:
                # Handle single item if it's not a list
                coerced = _coerce_entity_item(items)
                if coerced:
                    normalized[category] = [coerced]
        return normalized

    return entities

def prepare_and_clean_review_content(reviews: List[Dict]) -> str:
    """
    Combine title and review text from a list of reviews and clean HTML.
    """
    content_parts = []
    for review in reviews:
        text = review.get('reviewText', '').strip()
        title = review.get('summary', '').strip()

        if text or title:
            review_parts = []
            if title:
                review_parts.append(f"Title: {title}")
            if text:
                review_parts.append(f"Review: {text}")
            content_parts.append(' '.join(review_parts))

    content = ' '.join(content_parts)
    return clean_html_content(content)

def construct_user_preference_prompt(content: str, known_attributes: Optional[Dict] = None, product_info: Optional[Dict] = None) -> str:
    """
    Construct the detailed prompt for user preference extraction.
    """
    attributes_context = ""
    if known_attributes:
        lines = []
        for cat, vals in known_attributes.items():
            # Handle list or string values
            if isinstance(vals, list):
                val_str = ", ".join([str(v) for v in vals])
            else:
                val_str = str(vals)
            lines.append(f"- {cat}: {val_str}")
        if lines:
            attributes_context = "\n**Known Product Attributes (from Knowledge Base):**\n" + "\n".join(lines) + "\n\n"

    product_context = ""
    if product_info:
        title = product_info.get('title', '')
        desc = product_info.get('description', '')
        features = product_info.get('feature', [])
        price = product_info.get('price', '')
        details = product_info.get('details', {})
        
        info_parts = []
        if title: info_parts.append(f"**Product Title:** {title}")
        if price: info_parts.append(f"**Product Price:** {price}")
        if desc: info_parts.append(f"**Product Description:** {desc}")
        if features:
            if isinstance(features, list):
                f_list = [f"- {f}" for f in features]
                info_parts.append(f"**Product Features:**\n" + "\n".join(f_list))
            else:
                info_parts.append(f"**Product Features:** {features}")

        if details:
            details_str = ", ".join([f"{k}: {v}" for k, v in details.items() if v])
            if details_str: info_parts.append(f"**Product Details:** {details_str}")
        
        if info_parts:
            product_context = "\n**Product Unstructured Information:**\n" + "\n\n".join(info_parts) + "\n\n"

    prompt = f"""Analyze the user review below and identify mentions of product attributes.

{product_context}{attributes_context}**User Review Content:**
{content}

**Task:**
You are an expert product analyst. Your goal is to extract user preferences from the user reviews.

**Reasoning Process (Mental Scratchpad - Explicitly Perform These Steps):**

1.  **Identify Entities:** Extract all product limitations, features, qualities, or specifications mentioned by the user in the review.
2.  **Determine Sentiment:** Is the user's attitude towards this entity Positive, Neutral, or Negative?
3.  **Apply Filtering Rules (CRITICAL):**
    *   **Logic A: Negative Entities (COMPLAINTS):**
        *   **Action:** **ALWAYS KEEP**.
        *   **Reasoning:** If a user complains about something (Negative), it is a valid preference regardless of whether the product claims to have it.
        *   **Requirement:** You MUST provide an `improvement_wish`.
            *   *Explicit:* Use what they asked for (e.g., "wanted it smaller").
            *   *Implicit:* Infer the opposite (e.g., "too fragile" -> Wish: "Sturdy/Durable").
    *   **Logic B: Positive/Neutral Entities (VALIDATION):**
        *   **Action:** **CHECK SEMANTIC SIMILARITY**.
        *   **Test:** Does this entity semantically match ANY information in the **Known Product Attributes** OR **Product Unstructured Information** (Title/Description/Features)?
            *   *Yes (Match Found):* **KEEP IT**. This confirms the user noticed a real product feature.
            *   *No (No Match):* **DISCARD**. The user might be hallucinating or talking about something irrelevant to this specific product's data keys.

**Output Format:**
Create a JSON object with:
1. **"Product Category"**: (String) The most specific category from Know Product Attributes Category list.
2. **Standardized Category names** (as top-level keys for entities):
    - Each key maps to a list of objects containing:
        - "entity": The attribute value (Preference: match **Known Product Attributes** if possible, otherwise use the descriptive term).
        - "original_text": The exact text segment from the review.
        - "sentiment": "positive", "negative", or "neutral"
        - "improvement_wish": (String) **REQUIRED for Negative entities**.

Return ONLY the JSON object."""
    return prompt

def process_user_preference_extraction_response(response_str: str) -> tuple:
    """
    Process the LLM response to extract user preferences.
    Returns: (flattened_entities_list, categorized_entities_dict, extracted_min_category)
    """
    if not response_str:
        raise ExtractionException("No response from user preference extraction")

    try:
        # Clean the response
        response_str = response_str.strip()

        # Look for code blocks
        json_blocks = []
        start = 0
        while True:
            json_start = response_str.find('```json', start)
            if json_start == -1:
                break
            json_end = response_str.find('```', json_start + 7)
            if json_end == -1:
                break
            content_start = response_str.find('\n', json_start) + 1
            if content_start > 0:
                content = response_str[content_start:json_end].strip()
                if content:
                    json_blocks.append(content)
            start = json_end + 3

        # Also handle regular ``` blocks
        if not json_blocks:
            if '```' in response_str:
                last_triple = response_str.rfind('```')
                first_triple = response_str.rfind('```', 0, last_triple)
                if first_triple != last_triple:
                    content_start = response_str.find('\n', first_triple) + 1
                    if content_start > 0:
                        response_str = response_str[content_start:last_triple].strip()
                else:
                    content_start = response_str.find('\n', first_triple) + 1
                    if content_start > 0:
                        response_str = response_str[content_start:].strip()
        
        # Use the last json block if found
        if json_blocks:
            response_str = json_blocks[-1]

        # Try to parse as JSON
        try:
            result = json.loads(response_str)
        except json.JSONDecodeError:
            # Fallback: Extraction of JSON objects from messy text using brace matching
            potential_json_blocks = []
            stack = []
            start_idx = -1
            for i, char in enumerate(response_str):
                if char == '{':
                    if not stack:
                        start_idx = i
                    stack.append('{')
                elif char == '}':
                    if stack:
                        stack.pop()
                        if not stack and start_idx != -1:
                            potential_json_blocks.append(response_str[start_idx:i+1])
            
            if potential_json_blocks:
                success = False
                for block in reversed(potential_json_blocks):
                    try:
                        result = json.loads(block)
                        success = True
                        break
                    except:
                        continue
                if not success:
                    raise
            else:
                raise

        # Handle different response formats
        if isinstance(result, dict):
            # Extraction logic for entities
            categorized_entities = {}
            flattened = []
            extracted_min_category = result.get("Product Category")

            # Entities are usually everything except our special metadata keys
            for category, entities in result.items():
                if category in {"Product Category", "Reasoning"}:
                    continue
                if category.lower() in {"value", "sentiment"}:
                    continue
                
                category_norm = normalize_category_label(category)
                if isinstance(entities, list):
                    category_entities = []
                    seen_in_category = set()  # (entity_name, sentiment)
                    
                    for entity in entities:
                        normalized = _coerce_entity_item(entity)
                        if not normalized:
                            continue
                        
                        entity_text = normalized["entity"]
                        sentiment = normalized["sentiment"]
                        
                        # Deduplication logic
                        dedup_key = (entity_text.lower(), sentiment)
                        if dedup_key in seen_in_category:
                            continue
                        seen_in_category.add(dedup_key)

                        # Apply atomic filtering
                        entity_lower = entity_text.lower()
                        if (',' in entity_text or ' and ' in entity_lower or '&' in entity_text):
                            continue

                        category_entities.append(normalized)
                        flattened.append(entity_text)

                    if category_entities:
                        if category_norm not in categorized_entities:
                            categorized_entities[category_norm] = []
                        categorized_entities[category_norm].extend(category_entities)

            return flattened, categorized_entities, extracted_min_category
        else:
            raise ExtractionException(f"Invalid result format: expected dict, got {type(result)}")

    except Exception as e:
        raise ExtractionException(f"Failed to parse response: {str(e)}")

# CLI Interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="User Preference Extraction Skill")
    parser.add_argument("--mode", choices=["prompt", "parse"], required=True, help="Mode: generated prompt or parse response")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", help="Output JSON file path (optional, prints to stdout if not set)")
    parser.add_argument("--asin", help="Optional: specific ASIN to process (if input contains multiple products)")
    
    args = parser.parse_args()
    
    try:
        with open(args.input, 'r') as f:
            input_data = json.load(f)
            
        result = {}
        
        if args.mode == "prompt":
            # Check if input is aggregated list (new format) or single item (legacy)
            products = []
            if "products" in input_data and isinstance(input_data["products"], list):
                products = input_data["products"]
            else:
                # Treat as single item
                products = [input_data]
                
            # Filter by ASIN if requested
            if args.asin:
                products = [p for p in products if p.get("asin") == args.asin or p.get("target_asin") == args.asin]
                
            prompts_output = []
            for p in products:
                reviews = p.get("reviews", [])
                known_attributes = p.get("known_attributes", {})
                product_info = p.get("product_info", {})
                target_asin = p.get("asin") or p.get("target_asin", "unknown")
                
                content = prepare_and_clean_review_content(reviews)
                prompt_text = construct_user_preference_prompt(content, known_attributes, product_info)
                
                prompts_output.append({
                    "asin": target_asin,
                    "prompt": prompt_text
                })
            
            result = {"prompts": prompts_output}
            
        elif args.mode == "parse":
            response = input_data.get("response", "")
            flattened, categorized, min_cat = process_user_preference_extraction_response(response)
            result = {
                "flattened_entities": flattened,
                "categorized_entities": categorized,
                "min_category": min_cat
            }
            
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
