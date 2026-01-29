import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Make sibling modules (model, utils) importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import call_llm_with_retry, set_api_responses_file  # type: ignore
from utils import (  # type: ignore
    create_llm_with_config,
    get_all_api_keys_in_order,
    log_with_timestamp,
    try_api_keys_with_fallback,
)


# Input/Output paths
INPUT_PATH = "/home/wlia0047/ar57/wenyu/result/entity_matching_results.json"
OUTPUT_PATH = "/home/wlia0047/ar57/wenyu/result/entity_matching_results_with_std_dimensions.json"


@dataclass
class ParsedDimension:
    value: float
    unit: str  # canonical unit keyword, e.g., "mm", "inch", "yard"


def load_results(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(path: str, data: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def classify_product_type(title: str) -> str:
    """Rough product type from title keywords."""
    title_lower = title.lower()
    if any(k in title_lower for k in ["earring", "necklace", "ring", "jewelry", "bracelet"]):
        return "jewelry"
    if any(k in title_lower for k in ["floss", "thread", "yarn", "cord"]):
        return "thread"
    if any(k in title_lower for k in ["fabric", "cloth", "napkin", "towel", "sheet", "pillow", "case"]):
        return "fabric"
    return "general"


def _extract_numbers(text: str) -> List[float]:
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return [float(n) for n in numbers]


def _sanitize_spec_name(name: str) -> str:
    """
    Ensure the spec name is semantic text, not just a raw numeric value.

    Examples that should NOT be accepted as final spec names:
    - "16"
    - "16mm"
    - "14 x 20"
    """
    if not name:
        return "unknown_spec"

    stripped = str(name).strip()
    if not stripped:
        return "unknown_spec"

    # If the string is composed only of digits, spaces, punctuation, and unit-like letters,
    # treat it as non-semantic.
    compact = stripped.lower().replace(" ", "")
    if re.fullmatch(r"[0-9x\-\+\./\"'cmminftyd]+", compact or ""):
        return "unknown_spec"

    return stripped


def parse_dimension(text: str) -> Optional[ParsedDimension]:
    """Extract numeric value and unit from a dimension string."""
    lowered = text.lower()
    numbers = _extract_numbers(lowered)
    if not numbers:
        return None

    # Determine unit
    if any(u in lowered for u in ["mm", "millimeter"]):
        unit = "mm"
        value = max(numbers)
    elif any(u in lowered for u in ["cm", "centimeter"]):
        unit = "mm"
        value = max(numbers) * 10
    elif any(u in lowered for u in ["inch", "inches", '"', "in "]) and not any(
        u in lowered for u in ["yd", "yard", "yards"]
    ):
        unit = "mm"
        value = max(numbers) * 25.4
    elif any(u in lowered for u in ["ft", "foot", "feet"]):
        unit = "mm"
        value = max(numbers) * 304.8
    elif any(u in lowered for u in ["yard", "yards", "yd"]):
        unit = "yard"
        value = max(numbers)
    elif any(u in lowered for u in ["meter", "meters", "m "]):
        unit = "meter"
        value = max(numbers)
    else:
        # Default to millimeters
        unit = "mm"
        value = max(numbers)

    return ParsedDimension(value=value, unit=unit)


def bucket_dimension(prod_type: str, dim: ParsedDimension) -> str:
    """Map numeric value to a categorical spec (no digits)."""
    if prod_type == "jewelry":
        # Value already in mm
        if dim.value <= 10:
            return "petite size"
        if dim.value <= 25:
            return "standard size"
        return "statement size"

    if prod_type == "thread":
        # Prefer yards/meters as length; convert mm back to meters when needed
        if dim.unit == "yard":
            yard_val = dim.value
        elif dim.unit == "meter":
            yard_val = dim.value / 0.9144
        else:  # mm or other
            yard_val = dim.value / 914.4

        if yard_val < 5:
            return "short skein"
        if yard_val < 20:
            return "standard skein"
        return "bulk skein"

    if prod_type == "fabric":
        # Convert to mm; treat as length proxy
        mm_val = dim.value if dim.unit == "mm" else dim.value * 1000
        if mm_val < 300:
            return "compact cut"
        if mm_val < 800:
            return "medium cut"
        return "oversize cut"

    # General fallback
    if dim.value <= 30:
        return "compact size"
    if dim.value <= 100:
        return "mid size"
    return "large size"


def map_dimension_to_size(dimension_raw: str, spec_name: str) -> str:
    """
    Map dimension value to 大/中/小 based on parsed value and spec type.
    
    Args:
        dimension_raw: Raw dimension string (e.g., "800 Meters", "880 Yards")
        spec_name: Standardized spec name (e.g., "spool_length", "fabric_width")
    
    Returns:
        "大", "中", or "小"
    """
    parsed = parse_dimension(dimension_raw)
    if not parsed:
        return "中"  # Default to medium if cannot parse
    
    # Normalize to a common unit for comparison
    # For length-based specs, convert to meters
    # For diameter/width-based specs, convert to mm
    spec_lower = spec_name.lower()
    
    # Length-based specs (spool_length, fabric_length, chain_length, etc.)
    if "length" in spec_lower or "height" in spec_lower:
        if parsed.unit == "meter":
            value_m = parsed.value
        elif parsed.unit == "yard":
            value_m = parsed.value * 0.9144
        elif parsed.unit == "mm":
            value_m = parsed.value / 1000
        else:
            value_m = parsed.value
        
        # Thresholds for length (in meters)
        if value_m < 0.5:
            return "小"
        elif value_m < 2.0:
            return "中"
        else:
            return "大"
    
    # Diameter/width-based specs (diameter, width, etc.)
    elif "diameter" in spec_lower or "width" in spec_lower:
        if parsed.unit == "mm":
            value_mm = parsed.value
        elif parsed.unit == "meter":
            value_mm = parsed.value * 1000
        elif parsed.unit == "yard":
            value_mm = parsed.value * 914.4
        else:
            value_mm = parsed.value
        
        # Thresholds for diameter/width (in mm)
        if value_mm < 5:
            return "小"
        elif value_mm < 20:
            return "中"
        else:
            return "大"
    
    # Volume-based specs
    elif "volume" in spec_lower:
        # Assume volume is in ml or similar
        if parsed.value < 10:
            return "小"
        elif parsed.value < 50:
            return "中"
        else:
            return "大"
    
    # Package dimensions
    elif "package" in spec_lower:
        # Convert to cm for package dimensions
        if parsed.unit == "mm":
            value_cm = parsed.value / 10
        elif parsed.unit == "meter":
            value_cm = parsed.value * 100
        elif parsed.unit == "yard":
            value_cm = parsed.value * 91.44
        else:
            value_cm = parsed.value
        
        if value_cm < 10:
            return "小"
        elif value_cm < 30:
            return "中"
        else:
            return "大"
    
    # General fallback: use relative comparison based on numeric value
    # Normalize to a reasonable scale
    if parsed.unit == "mm":
        normalized_val = parsed.value
    elif parsed.unit == "meter":
        normalized_val = parsed.value * 1000
    elif parsed.unit == "yard":
        normalized_val = parsed.value * 914.4
    else:
        normalized_val = parsed.value
    
    # Use percentiles: assume values are distributed, use relative thresholds
    if normalized_val < 10:
        return "小"
    elif normalized_val < 100:
        return "中"
    else:
        return "大"


def normalize_dimensions(values: Iterable[str], prod_type: str) -> List[str]:
    normalized = []
    for val in values:
        parsed = parse_dimension(val)
        if not parsed:
            normalized.append("unmapped")
            continue
        normalized.append(bucket_dimension(prod_type, parsed))
    return normalized


def _build_product_context(item: Dict[str, Any], dimensions: List[str], prod_type: str) -> str:
    """Build a compact textual context for the LLM."""
    asin = item.get("asin", "Unknown")
    title = item.get("product_title") or item.get("product_info", {}).get("title") or ""
    product_info = item.get("product_info") or {}
    brand = product_info.get("brand", "")
    categories = product_info.get("category") or []
    features = product_info.get("feature") or []
    description = product_info.get("description") or []

    # Normalize to short text blocks
    if isinstance(categories, list):
        categories_text = ", ".join(str(c) for c in categories[:5])
    else:
        categories_text = str(categories)

    if isinstance(features, list):
        features_text = " ".join(str(f) for f in features[:5])
    else:
        features_text = str(features)

    if isinstance(description, list):
        desc_text = " ".join(str(d) for d in description[:3])
    else:
        desc_text = str(description)

    dims_text = [str(d) for d in dimensions]

    context = {
        "asin": asin,
        "product_type_hint": prod_type,
        "title": title,
        "brand": brand,
        "categories": categories_text,
        "features": features_text,
        "description": desc_text,
        "dimensions": dims_text,
    }
    return json.dumps(context, ensure_ascii=False)


def _parse_llm_dimension_response(raw: str, dimensions: List[str]) -> List[Dict[str, str]]:
    """
    Parse LLM response for dimensions.

    Expected final JSON format:
    [
      {"dimension_raw": "...", "spec_name": "..."},
      ...
    ]
    """
    if not raw:
        raise RuntimeError("Empty LLM response for dimension normalization")

    s = raw.strip()

    # Strip markdown code block if present
    if "```" in s:
        last_triple = s.rfind("```")
        first_triple = s.rfind("```", 0, last_triple)
        if first_triple != -1 and last_triple != -1 and first_triple != last_triple:
            content_start = s.find("\n", first_triple) + 1
            if content_start > 0:
                s = s[content_start:last_triple].strip()

    # Try to extract JSON array if surrounded by explanations
    if not s.lstrip().startswith("["):
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]

    try:
        parsed = json.loads(s)
    except Exception as e:
        raise RuntimeError(f"Failed to parse LLM dimension response as JSON: {e}") from e

    results: List[Dict[str, str]] = []

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            raw_dim = str(item.get("dimension_raw") or item.get("dimension") or "").strip()
            spec_name = _sanitize_spec_name(item.get("spec_name") or item.get("spec") or "")
            if not raw_dim:
                continue
            results.append({"dimension_raw": raw_dim, "spec_name": spec_name})

    # Ensure we have at least one valid entry
    if not results:
        raise RuntimeError("LLM dimension response contained no valid items")

    return results


def infer_dimensions_with_llm(
    item: Dict[str, Any], dimensions: List[str], prod_type: str, all_api_keys: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    Use LLM to infer semantic spec names for each raw dimension value.

    Output is a list of dicts:
    [
      {"dimension_raw": "...", "spec_name": "..."},
      ...
    ]

    spec_name MUST be semantic text (not bare numeric).
    """
    if not dimensions:
        return []

    context_json = _build_product_context(item, dimensions, prod_type)

    prompt = f"""
You are an expert in e-commerce product attribute standardization.

Given the following JSON describing ONE product and its raw dimension strings:
{context_json}

Step 1 – Think about SIZE TERMS for this product:
- First, understand what kind of product this is and what human-readable size terms are typically used
  to describe "small / medium / large" for this category.
- For example:
  - For bedding / mattresses / sheets: ["Twin", "Full", "Queen", "King", "California King"].
  - For clothing: ["XS", "S", "M", "L", "XL"].
  - For earrings / jewelry: ["small", "medium", "large", "extra large"] or more specific phrases.
- Infer a small set of such size terms that would make sense for THIS product type.

Step 2 – Map each numeric dimension to semantic SPEC + SIZE:
- For EACH raw dimension string in "dimensions", combine:
  (a) the numeric value(s) and unit(s), and
  (b) your inferred size terms for this product type,
  and decide whether this dimension corresponds to a relatively small / medium / large size
  and which size TERM (e.g., "Queen Size", "King Size", "Large Hoop", "Small Pendant") best describes it.
- Then decide which semantic spec dimension it represents (e.g., "fabric_width", "fabric_length",
  "chain_length", "package_height", "ring_inner_diameter", etc.).
- Your final spec_name should reflect both:
  - WHAT dimension it is (width/length/height/diameter/inner_diameter/chain_length/overall_length/etc.), and
  - Optionally the relative size tier or size term when helpful (e.g., "sheet_length_queen", "ring_inner_diameter_large").

IMPORTANT CONSTRAINTS:
- spec_name CANNOT be a pure numeric-like string or only number+unit
  (e.g., "16", "16mm", "14 x 20" are NOT valid spec_name).
- spec_name MUST be a semantic label describing WHAT the size refers to, such as:
  - "earring_diameter_small"
  - "earring_drop_length"
  - "fabric_width"
  - "fabric_length"
  - "chain_length"
  - "package_height"
  - "package_width"
  - "package_length"
  - "spool_length"
- It is allowed for spec_name to contain letters and numbers together
  (e.g., "fabric_width_for_20x30cm_mat" or "sheet_length_queen"),
  but it must clearly describe the kind of size, not be just the numeric string itself.
- If you really cannot identify a reasonable spec, use "unknown_spec" as spec_name.

OUTPUT FORMAT (JSON ONLY):
- Return a JSON array.
- Each element must have the form:
  {{"dimension_raw": "<one of the original dimension strings>", "spec_name": "<semantic spec name, not numeric-only>"}}
- The array should cover ALL given dimension strings.
- Do not include any comments or explanations outside the JSON.
"""

    def _operation(api_config, provider_name, key_index):
        llm_model = create_llm_with_config(api_config)
        # Only call LLM and save raw response to api_raw_responses.json.
        # Parsing is performed later in a second pass from the saved file.
        asin = item.get("asin")
        meta = {
            "asin": asin.strip().upper() if isinstance(asin, str) and asin.strip() else None,
            "dimensions": dimensions,
            "prod_type": prod_type,
        }
        _content, success = call_llm_with_retry(
            llm_model,
            prompt,
            context="dimension_normalization",
            meta=meta,
        )
        if not success:
            raise RuntimeError("LLM call failed for dimension_normalization")
        return []

    result, success = try_api_keys_with_fallback(
        all_api_keys,
        _operation,
        context=f"{item.get('asin', 'Unknown')} dimension normalization",
    )
    if success:
        return []
    raise RuntimeError(f"LLM dimension normalization failed for ASIN {item.get('asin', 'Unknown')}")


def _load_saved_dimension_normalization(
    api_responses_file: str,
) -> Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, str]]]:
    """
    Load saved raw responses and parse them into standardized dimensions.

    Returns mapping:
      (ASIN, tuple(dimensions)) -> parsed standardized dimension entries
    """
    if not os.path.exists(api_responses_file):
        log_with_timestamp(f"⚠️ API responses file not found: {api_responses_file}")
        return {}

    try:
        with open(api_responses_file, "r", encoding="utf-8") as f:
            all_responses = json.load(f)
    except Exception as e:
        log_with_timestamp(f"❌ Error reading API responses file: {e}")
        return {}

    filtered = [
        r
        for r in (all_responses or [])
        if r.get("context") == "dimension_normalization" and r.get("success", False)
    ]

    parsed: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, str]]] = {}
    for r in filtered:
        try:
            meta = r.get("meta") or {}
            asin = meta.get("asin")
            dims = meta.get("dimensions") or []
            if not (isinstance(asin, str) and asin.strip()):
                continue
            if not isinstance(dims, list):
                dims = [str(dims)]
            dims_clean = tuple(str(d).strip() for d in dims if str(d).strip())
            if not dims_clean:
                continue

            raw = r.get("raw_response") or {}
            content = raw.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue

            parsed[(asin.strip().upper(), dims_clean)] = _parse_llm_dimension_response(
                content, list(dims_clean)
            )
        except Exception:
            continue

    return parsed


def process(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure raw responses are saved like other modules.
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    result_dir = os.path.join(workspace_root, "result")
    os.makedirs(result_dir, exist_ok=True)
    api_responses_file = os.path.join(result_dir, "api_raw_responses.json")
    set_api_responses_file(api_responses_file)

    data = load_results(input_path)
    updated: list

    # Prepare API keys once for all products
    all_api_keys = get_all_api_keys_in_order()

    # Handle two possible JSON shapes:
    # 1) {"user_id": "...", "products": [ {...}, ... ]}
    # 2) [{...}, {...}, ...]
    if isinstance(data, dict):
        products = data.get("products", [])
        updated_products = []

        total = len(products)
        # Phase 1: issue LLM calls (raw saved to file) without parsing.
        for idx, item in enumerate(products, 1):
            product_entities = item.get("product_entities") or {}
            dimensions: List[str] = product_entities.get("Dimensions") or []
            prod_type = classify_product_type(item.get("product_title", ""))

            if dimensions:
                infer_dimensions_with_llm(item, dimensions, prod_type, all_api_keys)
            item["standardized_dimensions"] = []
            updated_products.append(item)

            if idx % 10 == 0 or idx == total:
                log_with_timestamp(f"📏 Dimension normalization progress: {idx}/{total} products processed")

        # Phase 2: parse saved raw responses and attach standardized dimensions.
        parsed_map = _load_saved_dimension_normalization(api_responses_file)
        for item in updated_products:
            product_entities = item.get("product_entities") or {}
            dimensions = product_entities.get("Dimensions") or []
            if not isinstance(dimensions, list):
                dimensions = [str(dimensions)]
            dims_clean = tuple(str(d).strip() for d in dimensions if str(d).strip())
            asin = item.get("asin")
            asin_key = asin.strip().upper() if isinstance(asin, str) and asin.strip() else ""
            standardized = parsed_map.get((asin_key, dims_clean), [])

            for dim_entry in standardized:
                dim_entry["dimension_size"] = map_dimension_to_size(
                    dim_entry.get("dimension_raw", ""),
                    dim_entry.get("spec_name", ""),
                )
            item["standardized_dimensions"] = standardized

        data["products"] = updated_products
        updated = data
    else:
        updated_items = []
        total = len(data)
        # Phase 1: issue LLM calls (raw saved to file) without parsing.
        for idx, item in enumerate(data, 1):
            product_entities = item.get("product_entities") or {}
            dimensions = product_entities.get("Dimensions") or []
            if not isinstance(dimensions, list):
                # Be defensive about unexpected formats
                dimensions = [str(dimensions)]

            prod_type = classify_product_type(item.get("product_title", ""))

            if dimensions:
                infer_dimensions_with_llm(item, dimensions, prod_type, all_api_keys)
            item["standardized_dimensions"] = []
            updated_items.append(item)

            if idx % 10 == 0 or idx == total:
                log_with_timestamp(f"📏 Dimension normalization progress: {idx}/{total} products processed")

        # Phase 2: parse saved raw responses and attach standardized dimensions.
        parsed_map = _load_saved_dimension_normalization(api_responses_file)
        for item in updated_items:
            product_entities = item.get("product_entities") or {}
            dimensions = product_entities.get("Dimensions") or []
            if not isinstance(dimensions, list):
                dimensions = [str(dimensions)]
            dims_clean = tuple(str(d).strip() for d in dimensions if str(d).strip())
            asin = item.get("asin")
            asin_key = asin.strip().upper() if isinstance(asin, str) and asin.strip() else ""
            standardized = parsed_map.get((asin_key, dims_clean), [])

            for dim_entry in standardized:
                dim_entry["dimension_size"] = map_dimension_to_size(
                    dim_entry.get("dimension_raw", ""),
                    dim_entry.get("spec_name", ""),
                )
            item["standardized_dimensions"] = standardized

        updated = updated_items

    save_results(output_path, updated)
    print(f"✅ Saved standardized dimensions to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()
    process(input_path=args.input, output_path=args.output)
