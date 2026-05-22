#!/usr/bin/env python3
"""
Generate 10 Baby_Products queries per user with attribute validation only.

This variant keeps the syntax-depth prompt context but does not validate the
generated query's dependency-tree depth. Every retained query must still use
exactly the five provided product attributes.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


sys.path.insert(0, "/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query")

BASE_SCRIPT = Path("/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/06_generate_by_persona_placeholder_Baby_Products.py")
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(f"base script not found: {BASE_SCRIPT}")

_BASE_SPEC = importlib.util.spec_from_file_location("stage6_baby_base", BASE_SCRIPT)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise ImportError(f"cannot load base script: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(base)


CATEGORY = "Baby_Products"
SYNTAX_DEPTH_FILE = Path(
    "/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis"
) / CATEGORY / "user_average_syntax_depth.json"
OUTPUT_FILE = Path(
    "/home/wlia0047/ar57/wenyu/result/personal_query/06_query"
) / CATEGORY / "query_by_syntax_depth_no_depth_check_10.json"

NUM_USERS_TO_TEST = base.NUM_USERS_TO_TEST
MAX_WORKERS = base.MAX_WORKERS
REQUIRED_ATTR_COUNT = base.REQUIRED_ATTR_COUNT
NUM_CANDIDATES_PER_USER = 10
CACHE_PREWARM_MARKER = "CACHE_PREWARM_ONLY"
SYNTAX_DEPTH_SYSTEM_BASE = f"""You generate e-commerce product search queries for {CATEGORY}.

Construction advice:
- First include the core product type and brand.
- Then place the remaining attributes into a natural first-person shopping request.
- Make candidates structurally different from each other.
- Do not generate near-duplicate surface rewrites of the same syntax shape.
- Do not repeat any attribute value.
- Keep every attribute value in its original surface form.
- Do not singularize, pluralize, stem, paraphrase, or partially rewrite any attribute value.
- If two attribute values are similar, such as singular/plural variants, you must still preserve each one exactly as given.
- Never collapse the query into a keyword list or telegraphic phrase. Every candidate must be a full natural sentence.

Critical lexical constraint:
- The user prompt provides exactly five Product attributes.
- Every candidate query must include all five provided attribute values.
- In each candidate query, each provided attribute value must appear exactly once.
- Copy each attribute value exactly as written.
- Do not change singular/plural form.
- Do not replace an attribute value with a shorter variant.
- Do not merge two similar attribute values into one mention.
- If an attribute value is awkward, still preserve it exactly and build grammar around it.

Candidate clause-count plan:
- Generate exactly 10 candidates.
- Candidates 1-2 must have no dependent clauses.
- Candidates 3-4 must have exactly one dependent clause.
- Candidates 5-6 must have exactly two dependent clauses.
- Candidates 7-8 must have exactly three dependent clauses.
- Candidates 9-10 must have exactly four dependent clauses.
- A dependent clause means a finite relative, complement, or subordinate clause, usually introduced by words such as "that", "which", "who", "when", "where", "because", "while", "if", or "as".
- Do not count the main clause as a dependent clause.
- Use the assigned clause-count group to create structurally diverse candidates.
- Within each pair, use different attachment patterns.
- Across all 10 candidates, vary clause type and attachment point.

Output schema:
- Return JSON only in this exact schema:
  {{"candidates": [{{"query": "..."}}]}}

Hard rules:
1. Output JSON only.
2. The query must be a natural first-person e-commerce search query.
3. Include exactly the five Product attribute values provided in the user prompt.
4. Each provided Product attribute value must appear exactly once in each candidate query.
5. Do not invent extra product facts.
6. Generate exactly the number of candidates requested by the user prompt.
7. If you cannot satisfy every rule, output {{"status":"IMPOSSIBLE","reason":"..."}}.
8. If the user prompt contains {CACHE_PREWARM_MARKER}, this is cache warmup only. In that case, return the requested number of placeholder candidates with "query": "none"."""


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _round_target_depth(avg_depth: float) -> int:
    if not isinstance(avg_depth, (int, float)):
        raise TypeError(f"avg_depth must be numeric, got {type(avg_depth).__name__}")
    if not math.isfinite(avg_depth):
        raise ValueError(f"avg_depth must be finite, got {avg_depth}")
    target_depth = int(round(avg_depth))
    if target_depth < 1:
        raise ValueError(f"target syntax depth must be >= 1, got {target_depth}")
    return target_depth


def load_user_syntax_depths() -> dict[str, dict]:
    if not SYNTAX_DEPTH_FILE.exists():
        raise FileNotFoundError(f"syntax depth file not found: {SYNTAX_DEPTH_FILE}")
    payload = json.loads(SYNTAX_DEPTH_FILE.read_text(encoding="utf-8"))
    users = payload.get("users")
    if not isinstance(users, list):
        raise TypeError("syntax depth file must contain a top-level users list")

    depth_map = {}
    for idx, user in enumerate(users):
        if not isinstance(user, dict):
            raise TypeError(f"users[{idx}] must be an object")
        user_id = user.get("user_id")
        avg_depth = user.get("avg_depth")
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"users[{idx}].user_id must be a non-empty string")
        if avg_depth is None:
            raise KeyError(f"users[{idx}] missing avg_depth")
        depth_map[user_id] = {
            "avg_depth": float(avg_depth),
            "target_depth": _round_target_depth(float(avg_depth)),
            "review_count": user.get("review_count"),
            "min_depth": user.get("min_depth"),
            "max_depth": user.get("max_depth"),
        }
    return depth_map

def build_syntax_depth_prompt(
    target_depth: int,
    avg_depth: float,
    attrs: dict,
    candidate_count: int = NUM_CANDIDATES_PER_USER,
) -> tuple[str, str]:
    _ = target_depth, avg_depth
    user_content = f"""Product attributes:
{base._format_attrs_for_prompt(attrs)}

Return exactly {candidate_count} candidates.
"""
    return SYNTAX_DEPTH_SYSTEM_BASE, user_content


def call_llm_no_empty_retry(prompt: str, system_base: str, step_name: str) -> str:
    if base._minimax_client is None:
        base.load_minimax_client()

    cache_info = {
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    if system_base and getattr(base, "_first_request", False):
        log(f"[Request] {step_name} system_base (FIRST REQUEST - cache creation):\n{system_base}")
        base._first_request = False

    log(f"[Request] {step_name} user_content:\n{prompt}")

    response, cache_info = base._minimax_client.call_with_cache(
        system_base=system_base,
        user_content=prompt,
        max_tokens=32768,
        temperature=0.8,
        retry_on_empty_response=False,
        stream=True,
    )

    if not response:
        log(f"[Cache] {cache_info}")
        log(f"[ERROR] {step_name} empty response, marked failed without retry")
        return ""

    log(f"[Cache] {cache_info}")
    log(f"[Response] {step_name} response:\n{response[:1500]}")
    return response


def build_syntax_depth_prewarm_prompt() -> tuple[str, str]:
    user_content = f"""{CACHE_PREWARM_MARKER}

This request is only for prompt-cache construction. Do not solve the real task.

Return JSON only in this exact schema:
{{"candidates": [{{"query": "none"}}]}}

Return exactly {NUM_CANDIDATES_PER_USER} candidates.
Every candidate must use:
- "query": "none"
"""
    return SYNTAX_DEPTH_SYSTEM_BASE, user_content


def prewarm_syntax_depth_cache() -> None:
    system_base, user_content = build_syntax_depth_prewarm_prompt()
    call_llm_no_empty_retry(user_content, system_base=system_base, step_name="SyntaxDepthCachePrewarm")


def parse_syntax_depth_response(text_content: str, target_depth: int) -> list[dict]:
    try:
        json_match = re.search(r"\{[\s\S]*\}", text_content)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(text_content)
    except Exception as exc:
        log(f"    [DEBUG] Syntax-depth JSON parse failed: {exc}")
        return []

    if not isinstance(data, dict):
        log("    [DEBUG] Syntax-depth response is not an object")
        return []
    if data.get("status") == "IMPOSSIBLE":
        log(f"    [DEBUG] Model returned IMPOSSIBLE: {data.get('reason')}")
        return []

    raw_candidates = data.get("candidates")
    if raw_candidates is None and "query" in data:
        raw_candidates = [data]
    if not isinstance(raw_candidates, list):
        log("    [DEBUG] Syntax-depth response missing candidates list")
        return []

    candidates = []
    for idx, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            log(f"    [DEBUG] candidates[{idx}] is not an object")
            continue
        query = candidate.get("query")
        if not isinstance(query, str) or not query.strip():
            log(f"    [DEBUG] candidates[{idx}] missing non-empty query")
            continue
        candidates.append(
            {
                "target_depth": target_depth,
                "query": query.strip(),
                "word_count": base.count_words(query.strip()),
            }
        )
    return candidates


def _load_json_file(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"required file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_user_tasks() -> list[dict]:
    syntax_depth_map = load_user_syntax_depths()
    attr_density_profiles = _load_json_file(base.ATTR_DENSITY_PROFILES_FILE)
    attr_values_data = _load_json_file(base.ATTR_VALUES_FILE)

    user_wpa_map = {}
    for profile in attr_density_profiles:
        uid = profile.get("user_id")
        wpa = profile.get("words_per_attribute")
        if isinstance(uid, str) and uid and wpa is not None:
            user_wpa_map[uid] = float(wpa)

    if not isinstance(attr_values_data, dict) or "products" not in attr_values_data:
        raise TypeError("attributes file must contain top-level products list")

    user_prod_map: dict[str, list[dict]] = {}
    for product in attr_values_data["products"]:
        if not isinstance(product, dict):
            raise TypeError("every product entry must be an object")
        uid = product.get("user_id")
        if isinstance(uid, str) and uid:
            user_prod_map.setdefault(uid, []).append(product)

    candidate_user_ids = sorted(set(syntax_depth_map) & set(user_prod_map) & set(user_wpa_map))
    log(f"syntax depth users: {len(syntax_depth_map)}")
    log(f"users with product attrs: {len(user_prod_map)}")
    log(f"users with attr density: {len(user_wpa_map)}")
    log(f"candidate users: {len(candidate_user_ids)}")

    completed_user_ids = set()
    existing_results = []
    if OUTPUT_FILE.exists():
        existing_results = _load_json_file(OUTPUT_FILE)
        if not isinstance(existing_results, list):
            raise TypeError(f"existing output must be a list: {OUTPUT_FILE}")
        completed_user_ids = {item["user_id"] for item in existing_results if "user_id" in item}
        log(f"existing completed users: {len(completed_user_ids)}")

    tasks = []
    skipped_attr_count = 0
    for uid in candidate_user_ids:
        if uid in completed_user_ids:
            continue
        product = user_prod_map[uid][0]
        attrs = base._extract_attrs_from_product(product)
        if len(attrs) < REQUIRED_ATTR_COUNT:
            skipped_attr_count += 1
            continue
        depth_info = syntax_depth_map[uid]
        tasks.append(
            {
                "user_id": uid,
                "asin": product.get("asin", ""),
                "attrs": attrs,
                "syntax_depth": depth_info,
                "words_per_attribute": user_wpa_map[uid],
            }
        )
        if len(tasks) >= NUM_USERS_TO_TEST:
            break

    log(f"skipped by attr count: {skipped_attr_count}")
    log(f"new tasks: {len(tasks)}")
    return existing_results, tasks


def process_one_user(task: dict) -> dict | None:
    uid = task["user_id"]
    attrs = task["attrs"]
    source_attrs_used = base._attrs_used_from_source(attrs)
    avg_depth = task["syntax_depth"]["avg_depth"]
    target_depth = task["syntax_depth"]["target_depth"]

    system_base, user_content = build_syntax_depth_prompt(target_depth, avg_depth, attrs)
    response = call_llm_no_empty_retry(user_content, system_base=system_base, step_name="SyntaxDepthQuery")
    if not response:
        log(f"    [ERROR] Empty response, user={uid}")
        return None
    candidates = parse_syntax_depth_response(response, target_depth)
    if not candidates:
        log(f"    [ERROR] Syntax-depth query parse failed, user={uid}")
        return None

    rejection_reasons = []
    accepted_queries = []
    for idx, parsed in enumerate(candidates, start=1):
        query = parsed["query"]
        attr_ok, attr_error = base.validate_query_uses_exactly_five_attrs(query, source_attrs_used)
        if not attr_ok:
            rejection_reasons.append(f"candidate={idx}: attr failed: {attr_error}")
            continue

        accepted_queries.append(
            {
                "target_depth": target_depth,
                "actual_depth": None,
                "depth_validation_skipped": True,
                "user_avg_depth": avg_depth,
                "query": query,
                "word_count": base.count_words(query),
                "attrs_used": dict(source_attrs_used),
                "accepted_candidate_index": idx,
                "candidate_count": len(candidates),
            }
        )

    if len(accepted_queries) != NUM_CANDIDATES_PER_USER:
        log(
            f"    [ERROR] Expected {NUM_CANDIDATES_PER_USER} attr-valid candidates, "
            f"user={uid}, accepted={len(accepted_queries)}, candidates={len(candidates)}"
        )
        for reason in rejection_reasons[:5]:
            log(f"      [REJECT] {reason}")
        return None

    return {
        "user_id": uid,
        "asin": task["asin"],
        "syntax_depth_queries": accepted_queries,
        "syntax_depth_query": accepted_queries[0],
        "query_count": len(accepted_queries),
        "depth_validation_skipped": True,
    }


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def main() -> None:
    existing_results, tasks = build_user_tasks()
    if not tasks:
        raise ValueError("No syntax-depth query tasks to process")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    base.load_minimax_client()
    prewarm_syntax_depth_cache()

    results = existing_results.copy()
    existing_completed_before_run = len(existing_results)
    run_success_count = 0
    failed_users = []
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(process_one_user, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()
            if result:
                results.append(result)
                run_success_count += 1
                write_json_atomic(OUTPUT_FILE, results)
                log(
                    f"  [run_success={run_success_count}/{len(tasks)}] "
                    f"total_records={len(results)} user={result['user_id'][:20]}"
                )
            else:
                failed_users.append(task["user_id"])

    elapsed = time.time() - total_start
    log("=" * 60)
    log(f"existing completed users before run: {existing_completed_before_run}")
    log(f"successful users in this run: {run_success_count}")
    log(f"failed users in this run: {len(failed_users)}")
    log(f"processed users in this run: {len(tasks)}")
    log(f"total records after merge: {len(results)}")
    log(f"elapsed: {elapsed:.1f}s")
    log(f"output={OUTPUT_FILE}")


if __name__ == "__main__":
    main()
