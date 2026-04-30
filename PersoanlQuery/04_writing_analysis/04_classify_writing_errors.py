#!/usr/bin/env python3
"""Use Stage 5 syntax cache to classify writing errors as ACL/CCOMP.

Input:
  /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/{category}/writing_error.json

Output:
  /home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis/{category}/acl_ccomp_error.json
"""

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CATEGORIES = [
    "Baby_Products",
    "Grocery_and_Gourmet_Food",
    "Pet_Supplies",
]
WRITING_ANALYSIS_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/04_writing_analysis")
STAGE1_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction")
SYNTACTIC_ANALYSIS_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/05_syntactic_analysis")
PROGRESS_INTERVAL_USERS = 25


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def normalize_space(text: str) -> str:
    return " ".join(text.split())


# 常见英语词缀
COMMON_AFFIXES = ['s', 'es', 'ed', 'ing', 'er', 'est', 'ly', 'd', 'en', 'n']


def is_affix_variation(original: str, corrected: str) -> bool:
    """检查两个词是否仅仅是词缀变化（如 dog->dogs, jump->jumped）
    或者仅仅相差一个单引号（如 Its->It's）
    或者编辑距离很小的简单词汇错误（如 creat->create, an->a, to->too）
    或者仅仅是标点差异（如 a.->a）
    或者仅仅是单复数/时态/元音变换等简单错误
    """
    orig = original.lower().strip()
    corr = corrected.lower().strip()

    if orig == corr:
        return False

    # 过滤仅相差标点的错误（如 a. -> a）
    import string
    PUNCT = set(string.punctuation)
    orig_no_punct = ''.join(c for c in orig if c not in PUNCT)
    corr_no_punct = ''.join(c for c in corr if c not in PUNCT)
    if orig_no_punct == corr_no_punct and orig != corr:
        return True

    # 过滤仅相差单引号的错误（如 Its -> It's, dont -> don't）
    orig_no_quote = orig.replace("'", "")
    corr_no_quote = corr.replace("'", "")
    if orig_no_quote == corr_no_quote and orig != corr:
        return True

    # 计算编辑距离
    def edit_distance(s1: str, s2: str) -> int:
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    # 过滤编辑距离 <= 2 的简单词汇错误（如 creat->create, an->a, to->too, was->were）
    # 这些不是真正的 ACL/CCOMP 语法错误，而是简单的拼写/形态错误
    if len(orig) >= 2 and len(corr) >= 2:
        dist = edit_distance(orig, corr)
        if dist <= 2:
            return True

    # 过滤长度相差1且所有字符都在另一个词中的情况（如 a->an）
    if abs(len(orig) - len(corr)) == 1:
        shorter, longer = (orig, corr) if len(orig) < len(corr) else (corr, orig)
        if all(c in longer for c in shorter):
            return True

    # 过滤常见的主谓不一致/动词形式错误（如 was->were, is->are, do->did）
    COMMON_VERB_VARIATIONS = {
        ('was', 'were'), ('is', 'are'), ('are', 'is'),
        ('do', 'did'), ('does', 'did'),
    }
    if (orig, corr) in COMMON_VERB_VARIATIONS or (corr, orig) in COMMON_VERB_VARIATIONS:
        return True

    # 过滤常见的人称代词变化（如 I->me, me->I, he->him 等）
    COMMON_PRONOUN_VARIATIONS = {
        ('i', 'me'), ('me', 'i'),
        ('he', 'him'), ('him', 'he'),
        ('she', 'her'), ('her', 'she'),
        ('we', 'us'), ('us', 'we'),
        ('they', 'them'), ('them', 'they'),
    }
    if (orig, corr) in COMMON_PRONOUN_VARIATIONS or (corr, orig) in COMMON_PRONOUN_VARIATIONS:
        return True

    # 确保 orig 是较短的词
    if len(orig) > len(corr):
        orig, corr = corr, orig

    # 词缀差异至少应该是1个字符，且较短词长度至少为3
    if len(corr) - len(orig) < 1 or len(orig) < 3:
        return False

    # 检查 corr 是否由 orig + 词缀 组成
    for affix in COMMON_AFFIXES:
        if corr == orig + affix:
            return True

    return False


def validate_word_error(original: str, corrected: str) -> Tuple[bool, str]:
    orig = original.lower().strip()
    corr = corrected.lower().strip()
    if not orig or not corr:
        return False, "empty_error"
    if orig == corr:
        return False, "case_or_identity_error"
    if len(orig.split()) != 1 or len(corr.split()) != 1:
        return False, "non_single_word_error"
    if is_affix_variation(original, corrected):
        return False, "affix_only_error"
    return True, "valid"


def get_category_paths(category: str) -> Tuple[Path, Path, Path]:
    writing_error_file = WRITING_ANALYSIS_ROOT / category / "writing_error.json"
    stage1_reviews_file = STAGE1_ROOT / category / "stage1_filtered_users_reviews.json"
    output_file = WRITING_ANALYSIS_ROOT / category / "acl_ccomp_error.json"
    return writing_error_file, stage1_reviews_file, output_file


def get_syntax_cache_paths(category: str) -> Tuple[Path, Path]:
    category_dir = SYNTACTIC_ANALYSIS_ROOT / category
    return category_dir / "acl_sentences.jsonl", category_dir / "ccomp_sentences.jsonl"


def cache_key(user_id: str, review_index: int) -> Tuple[str, int]:
    return user_id, review_index


def load_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required Stage 5 syntax cache file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def load_syntax_cache(category: str) -> Dict[Tuple[str, int], Dict[str, List[Dict]]]:
    acl_file, ccomp_file = get_syntax_cache_paths(category)
    syntax_cache: Dict[Tuple[str, int], Dict[str, List[Dict]]] = {}

    log(f"[{category}] Loading Stage 5 ACL cache from: {acl_file}")
    acl_rows = 0
    for row in load_jsonl(acl_file):
        if "user_id" not in row or "review_index" not in row or "acl_info" not in row:
            raise ValueError(f"Invalid ACL syntax cache row: {row}")
        key = cache_key(row["user_id"], row["review_index"])
        syntax_cache.setdefault(key, {"acl_info": [], "ccomp_info": []})["acl_info"] = row["acl_info"]
        acl_rows += 1

    log(f"[{category}] Loading Stage 5 CCOMP cache from: {ccomp_file}")
    ccomp_rows = 0
    for row in load_jsonl(ccomp_file):
        if "user_id" not in row or "review_index" not in row or "ccomp_info" not in row:
            raise ValueError(f"Invalid CCOMP syntax cache row: {row}")
        key = cache_key(row["user_id"], row["review_index"])
        syntax_cache.setdefault(key, {"acl_info": [], "ccomp_info": []})["ccomp_info"] = row["ccomp_info"]
        ccomp_rows += 1

    log(f"[{category}] Loaded Stage 5 syntax cache: acl_rows={acl_rows}, ccomp_rows={ccomp_rows}, merged_reviews={len(syntax_cache)}")
    return syntax_cache


def load_review_map(stage1_reviews_file: Path) -> Dict[str, List[str]]:
    if not stage1_reviews_file.exists():
        raise FileNotFoundError(f"Stage 1 reviews file not found: {stage1_reviews_file}")
    with stage1_reviews_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "users" not in data or not isinstance(data["users"], list):
        raise ValueError(f"Invalid Stage 1 review file structure: {stage1_reviews_file}")

    review_map = {}
    for user in data["users"]:
        user_id = user.get("user_id")
        if not user_id:
            continue
        flattened_reviews = []
        for product in user.get("results", []):
            for review_text in product.get("target_reviews", []):
                if isinstance(review_text, str):
                    flattened_reviews.append(review_text)
        review_map[user_id] = flattened_reviews
    return review_map


def format_elapsed(start_time: float) -> str:
    elapsed_seconds = int(time.time() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_review_text(review_map: Dict[str, List[str]], user_id: str, review_index: int) -> str:
    if user_id not in review_map:
        raise ValueError(f"User {user_id} not found in Stage 1 review map")
    reviews = review_map[user_id]
    if review_index < 0 or review_index >= len(reviews):
        raise ValueError(f"Review index {review_index} out of range for user {user_id}")
    return reviews[review_index]


def find_text_span(text: str, pattern_text: str) -> Optional[Tuple[int, int]]:
    if not pattern_text:
        return None
    match = re.search(re.escape(pattern_text), text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start(), match.end()


def locate_error_char_span(review_text: str, span_text: str, original: str) -> Tuple[Optional[Tuple[int, int]], str]:
    span = find_text_span(review_text, normalize_space(span_text))
    if span is not None:
        span_start, span_end = span
        original_match = re.search(re.escape(original), review_text[span_start:span_end], flags=re.IGNORECASE)
        if original_match is not None:
            return (span_start + original_match.start(), span_start + original_match.end()), "ok"
        return (span_start, span_end), "ok"

    pattern = re.compile(rf"\b{re.escape(original)}\b", flags=re.IGNORECASE)
    match = pattern.search(review_text)
    if match is None:
        return None, "syntax_original_not_found"
    return (match.start(), match.end()), "ok"


def is_modal_word(word: str) -> bool:
    return word.lower().strip() in {"can", "could", "may", "might", "must", "shall", "should", "will", "would"}


def validate_sentence_metadata(info: Dict) -> None:
    required_fields = (
        "sentence_start_char",
        "sentence_end_char",
        "position_char",
    )
    for field in required_fields:
        if field not in info:
            raise ValueError(f"Stage 5 syntax cache missing {field}: {info}")
        if not isinstance(info[field], int):
            raise ValueError(f"Stage 5 syntax cache has invalid {field}: {info}")


def sentence_level_char_match(error_span: Tuple[int, int], infos: List[Dict]) -> Optional[Dict]:
    if not infos:
        return None

    error_start, error_end = error_span
    positioned_infos = []
    for info in infos:
        validate_sentence_metadata(info)
        if info["sentence_start_char"] <= error_start and error_end <= info["sentence_end_char"]:
            positioned_infos.append(info)
    if not positioned_infos:
        return None
    return min(
        positioned_infos,
        key=lambda info: abs(info["position_char"] - error_start),
    )


def classify_from_stage5_cache(
    user_id: str,
    review_index: int,
    review_text: str,
    span_text: str,
    original: str,
    corrected: str,
    syntax_cache: Dict[Tuple[str, int], Dict[str, List[Dict]]],
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    key = cache_key(user_id, review_index)
    if key not in syntax_cache:
        raise ValueError(f"Stage 5 syntax cache missing review for user={user_id}, review_index={review_index}")

    error_span, reason = locate_error_char_span(review_text, span_text, original)
    if error_span is None:
        return None, None, None, reason

    syntax_info = syntax_cache[key]
    # CCOMP: 仅匹配 comp_type == 'ccomp'，排除 mark_advcl 等其他类型
    ccomp_match = sentence_level_char_match(error_span, syntax_info["ccomp_info"])
    if ccomp_match is not None:
        comp_type = ccomp_match.get("comp_type")
        if comp_type == "ccomp":
            if is_modal_word(original) or is_modal_word(corrected):
                return "ccomp", "modal", "modal_distortion", "ok"
            return "ccomp", "ccomp", "clause_boundary_error", "ok"

    # ACL: 仅匹配 acl 和 relcl_reference，排除 relcl_non_reference
    acl_match = sentence_level_char_match(error_span, syntax_info["acl_info"])
    if acl_match is not None:
        acl_type = acl_match.get("acl_type")
        if acl_type not in ("acl", "relcl_reference"):
            # relcl_non_reference 不归类为 ACL，继续检查 CCOMP
            pass
        else:
            region_type = "relcl" if acl_type == "relcl_reference" else "acl"
            if original.lower().strip().rstrip("s") == corrected.lower().strip().rstrip("s"):
                error_type = "np_inflection"
            else:
                error_type = "modifier_typo"
            return "acl", region_type, error_type, "ok"

    return None, None, None, "syntax_no_acl_ccomp_match"


def classify_detail(
    user_id: str,
    detail: Dict,
    review_map: Dict[str, List[str]],
    syntax_cache: Dict[Tuple[str, int], Dict[str, List[Dict]]],
) -> Tuple[List[Dict], Counter]:
    if "asin" not in detail:
        raise ValueError(f"Missing asin in detail: {detail}")
    if "review_index" not in detail:
        raise ValueError(f"Missing review_index in detail: {detail}")
    if "errors" not in detail or not isinstance(detail["errors"], list):
        raise ValueError(f"Missing errors list in detail: {detail}")

    review_text = get_review_text(review_map, user_id, detail["review_index"])
    grouped: Dict[Tuple[str, str, str, str], List[Dict]] = {}
    filtered_counts = Counter()
    for error in detail["errors"]:
        for required_key in ("original", "corrected", "confidence"):
            if required_key not in error:
                raise ValueError(f"Missing {required_key} in error: {error}")

        is_valid, reason = validate_word_error(error["original"], error["corrected"])
        if not is_valid:
            filtered_counts[reason] += 1
            continue

        span_text = error.get("span_text", "")
        category, region_type, error_type, reason = classify_from_stage5_cache(
            user_id=user_id,
            review_index=detail["review_index"],
            review_text=review_text,
            span_text=span_text,
            original=error["original"],
            corrected=error["corrected"],
            syntax_cache=syntax_cache,
        )
        if category is None:
            filtered_counts[reason] += 1
            continue

        key = (category, region_type, error_type, span_text)
        grouped.setdefault(key, []).append({
            "original": error["original"],
            "corrected": error["corrected"],
            "error_type": error_type,
            "confidence": error["confidence"],
        })

    classified_details = []
    for (category, region_type, error_type, span_text), errors in grouped.items():
        item = {
            "asin": detail["asin"],
            "review_index": detail["review_index"],
            "error_category": category,
            "region_type": region_type,
            "error_type": error_type,
            "errors": errors,
        }
        if span_text:
            item["span_text"] = span_text
        classified_details.append(item)
    return classified_details, filtered_counts


def classify_user(
    row: Dict,
    review_map: Dict[str, List[str]],
    syntax_cache: Dict[Tuple[str, int], Dict[str, List[Dict]]],
) -> Dict:
    for key in ("user_id", "status", "reviews_processed", "detailed_results"):
        if key not in row:
            raise ValueError(f"Missing {key} in row: {row}")
    if not isinstance(row["detailed_results"], list):
        raise ValueError(f"detailed_results must be a list for user {row['user_id']}")

    acl_error_types = Counter()
    ccomp_error_types = Counter()
    acl_region_types = Counter()
    ccomp_region_types = Counter()
    filtered_counts = Counter()
    detailed_results = []

    for detail in row["detailed_results"]:
        classified_details, detail_filtered_counts = classify_detail(row["user_id"], detail, review_map, syntax_cache)
        filtered_counts.update(detail_filtered_counts)
        for classified_detail in classified_details:
            count = len(classified_detail["errors"])
            if classified_detail["error_category"] == "acl":
                acl_error_types[classified_detail["error_type"]] += count
                acl_region_types[classified_detail["region_type"]] += count
            elif classified_detail["error_category"] == "ccomp":
                ccomp_error_types[classified_detail["error_type"]] += count
                ccomp_region_types[classified_detail["region_type"]] += count
            else:
                raise ValueError(f"Unsupported error_category: {classified_detail['error_category']}")
            detailed_results.append(classified_detail)

    acl_count = sum(acl_error_types.values())
    ccomp_count = sum(ccomp_error_types.values())
    return {
        "user_id": row["user_id"],
        "status": row["status"],
        "reviews_processed": row["reviews_processed"],
        "acl_error_count": acl_count,
        "ccomp_error_count": ccomp_count,
        "total_errors": acl_count + ccomp_count,
        "acl_error_types": dict(acl_error_types),
        "ccomp_error_types": dict(ccomp_error_types),
        "acl_region_types": dict(acl_region_types),
        "ccomp_region_types": dict(ccomp_region_types),
        "filtered_counts": dict(filtered_counts),
        "detailed_results": detailed_results,
    }


def user_has_errors(row: Dict) -> bool:
    detailed_results = row.get("detailed_results")
    if not isinstance(detailed_results, list):
        return False
    for detail in detailed_results:
        errors = detail.get("errors") if isinstance(detail, dict) else None
        if isinstance(errors, list) and errors:
            return True
    return False


def empty_classified_user(row: Dict) -> Dict:
    for key in ("user_id", "status", "reviews_processed", "detailed_results"):
        if key not in row:
            raise ValueError(f"Missing {key} in row: {row}")
    if not isinstance(row["detailed_results"], list):
        raise ValueError(f"detailed_results must be a list for user {row['user_id']}")
    return {
        "user_id": row["user_id"],
        "status": row["status"],
        "reviews_processed": row["reviews_processed"],
        "acl_error_count": 0,
        "ccomp_error_count": 0,
        "total_errors": 0,
        "acl_error_types": {},
        "ccomp_error_types": {},
        "acl_region_types": {},
        "ccomp_region_types": {},
        "filtered_counts": {},
        "detailed_results": [],
    }


def classify_category(category: str) -> None:
    start_time = time.time()
    writing_error_file, stage1_reviews_file, output_file = get_category_paths(category)
    log(f"=== Classifying category: {category} ===")
    log(f"[{category}] Reading writing errors from: {writing_error_file}")

    if not writing_error_file.exists():
        raise FileNotFoundError(f"Writing error file not found: {writing_error_file}")

    with writing_error_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Input must be a JSON list: {writing_error_file}")
    log(f"[{category}] Loaded {len(rows)} users from writing_error.json")
    rows_with_errors = [row for row in rows if user_has_errors(row)]
    rows_without_errors = len(rows) - len(rows_with_errors)
    log(
        f"[{category}] Users requiring syntax parsing: {len(rows_with_errors)}; "
        f"users skipped with no errors: {rows_without_errors}"
    )

    log(f"[{category}] Loading Stage 1 review map from: {stage1_reviews_file}")
    review_map = load_review_map(stage1_reviews_file)
    log(f"[{category}] Loaded review map for {len(review_map)} users")
    syntax_cache = load_syntax_cache(category)

    output_rows = []
    running_acl = 0
    running_ccomp = 0
    running_filtered_counts = Counter()
    parsed_users = 0
    total_parse_users = len(rows_with_errors)
    for index, row in enumerate(rows, start=1):
        did_parse = False
        if user_has_errors(row):
            output_row = classify_user(row, review_map, syntax_cache)
            parsed_users += 1
            did_parse = True
        else:
            output_row = empty_classified_user(row)
        output_rows.append(output_row)
        running_acl += output_row["acl_error_count"]
        running_ccomp += output_row["ccomp_error_count"]
        running_filtered_counts.update(output_row["filtered_counts"])

        if (
            total_parse_users == 0
            or (did_parse and parsed_users == 1)
            or (did_parse and parsed_users % PROGRESS_INTERVAL_USERS == 0)
            or index == len(rows)
        ):
            parsed_percent = parsed_users / total_parse_users if total_parse_users else 1.0
            log(
                f"[{category}] Parsed {parsed_users}/{total_parse_users} users with errors "
                f"({parsed_percent:.2%}); "
                f"scanned={index}/{len(rows)}; skipped_no_errors={index - parsed_users}; "
                f"ACL={running_acl}; CCOMP={running_ccomp}; "
                f"filtered={sum(running_filtered_counts.values())}; "
                f"elapsed={format_elapsed(start_time)}"
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_rows_with_errors = [row for row in output_rows if row["total_errors"] > 0]
    users_without_acl_ccomp_errors = len(output_rows) - len(output_rows_with_errors)
    log(f"[{category}] Writing classified output to: {output_file}")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(output_rows_with_errors, f, ensure_ascii=False, indent=2)

    total_acl = sum(row["acl_error_count"] for row in output_rows)
    total_ccomp = sum(row["ccomp_error_count"] for row in output_rows)
    filtered_counts = Counter()
    for row in output_rows:
        filtered_counts.update(row["filtered_counts"])
    users_with_errors = sum(1 for row in output_rows if row["total_errors"] > 0)

    log(f"Input users: {len(rows)}")
    log(f"Users with raw writing errors: {len(rows_with_errors)}")
    log(f"Users without raw writing errors: {rows_without_errors}")
    log(f"Users with ACL/CCOMP errors: {users_with_errors}")
    log(f"Users omitted from output without ACL/CCOMP errors: {users_without_acl_ccomp_errors}")
    log(f"ACL errors: {total_acl}")
    log(f"CCOMP errors: {total_ccomp}")
    if filtered_counts:
        log(f"Filtered outputs: {dict(filtered_counts)}")
    log(f"Output written to: {output_file}")
    log(f"=== Finished category: {category}; elapsed={format_elapsed(start_time)} ===")


def main() -> None:
    for category in CATEGORIES:
        classify_category(category)


if __name__ == "__main__":
    main()
