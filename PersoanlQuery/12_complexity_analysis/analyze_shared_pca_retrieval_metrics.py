#!/usr/bin/env python3
"""Connect shared query PCA scores with retrieval metrics for Baby_Products."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import kruskal, pearsonr, spearmanr


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
QUERY_PCA_FILE = INPUT_DIR / "review_query_shared_pca_query_records.jsonl"
RETRIEVAL_FILE = REPO_ROOT / "result" / "personal_query" / "08_retrieval" / CATEGORY / "retrieval_syntax_depth_summary.json"
SUMMARY_FILE = INPUT_DIR / "shared_pca_retrieval_metrics_summary.json"
RECORD_FILE = INPUT_DIR / "shared_pca_retrieval_metrics_records.jsonl"

METRIC_COLUMNS = ["hit_at10", "n_at10", "mrr_at10"]
TIER_LABELS = ["low", "medium", "high"]


def log(message: str) -> None:
    print(message, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} 为空")
    return rows


def summarize_array(values: np.ndarray) -> dict:
    if len(values) == 0:
        raise ValueError("无法汇总空数组")
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def query_key(user_id: str, asin: str) -> tuple[str, str]:
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id 必须是非空字符串")
    if not isinstance(asin, str) or not asin:
        raise ValueError("asin 必须是非空字符串")
    return user_id, asin


def load_query_pca_index() -> dict[tuple[str, str], dict]:
    rows = load_jsonl(QUERY_PCA_FILE)
    index: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = query_key(row["user_id"], row["asin"])
        if key in index:
            raise ValueError(f"query PCA 记录存在重复 key: {key}")
        index[key] = row
    return index


def load_aligned_records(query_index: dict[tuple[str, str], dict]) -> list[dict]:
    payload = json.loads(RETRIEVAL_FILE.read_text(encoding="utf-8"))
    combined = payload.get("all_results_combined")
    if not isinstance(combined, list) or not combined:
        raise ValueError("retrieval summary 缺少 all_results_combined")

    records = []
    missing_keys = set()
    seen_pairs = set()
    for retriever_result in combined:
        retriever = retriever_result.get("retriever")
        if not isinstance(retriever, str) or not retriever:
            raise ValueError("retriever 为空")
        query_records = retriever_result.get("all_query_records")
        if not isinstance(query_records, list) or not query_records:
            raise ValueError(f"{retriever} 缺少 all_query_records")

        for row in query_records:
            key = query_key(row["user_id"], row["asin"])
            pca_row = query_index.get(key)
            if pca_row is None:
                missing_keys.add(key)
                continue

            record_key = (retriever, key[0], key[1])
            if record_key in seen_pairs:
                raise ValueError(f"retrieval 记录重复: {record_key}")
            seen_pairs.add(record_key)

            record = {
                "domain": CATEGORY,
                "retriever": retriever,
                "user_id": row["user_id"],
                "asin": row["asin"],
                "target_depth": int(row["target_depth"]),
                "actual_depth": int(row["syntax_depth"]),
                "shared_pca_score": float(pca_row["shared_pca_score"]),
                "query_length": float(row["query_length"]),
                "avg_idf": float(row["mean_idf"]),
                "hit_at10": float(row["hit_at10"]),
                "n_at10": float(row["n_at10"]),
                "mrr_at10": float(row["mrr_at10"]),
            }
            records.append(record)

    if missing_keys:
        raise ValueError(f"retrieval 中有 {len(missing_keys)} 个 query key 缺少 PCA 记录")
    if not records:
        raise ValueError("没有成功对齐的 retrieval/PCA 记录")
    return records


def assign_pca_tiers(records: list[dict]) -> dict:
    unique_scores_by_key = {}
    for row in records:
        key = query_key(row["user_id"], row["asin"])
        score = float(row["shared_pca_score"])
        existing = unique_scores_by_key.get(key)
        if existing is not None and existing != score:
            raise ValueError(f"同一 query key 出现不同 PCA score: {key}")
        unique_scores_by_key[key] = score

    score_array = np.array(list(unique_scores_by_key.values()), dtype=float)
    q33 = float(np.quantile(score_array, 1.0 / 3.0))
    q67 = float(np.quantile(score_array, 2.0 / 3.0))
    if not q33 < q67:
        raise ValueError("PCA 三分位边界异常")

    tier_by_key = {}
    for key, score in unique_scores_by_key.items():
        if score <= q33:
            tier = "low"
        elif score <= q67:
            tier = "medium"
        else:
            tier = "high"
        tier_by_key[key] = tier

    for row in records:
        row["pca_tier"] = tier_by_key[query_key(row["user_id"], row["asin"])]

    return {"q33": q33, "q67": q67}


def metric_group_summary(rows: list[dict]) -> dict:
    summary = {"count": int(len(rows))}
    for metric in METRIC_COLUMNS:
        values = np.array([float(row[metric]) for row in rows], dtype=float)
        summary[metric] = summarize_array(values)
    return summary


def build_tier_analysis(records: list[dict]) -> dict:
    output = {}
    retrievers = sorted({row["retriever"] for row in records})
    scopes = ["pooled"] + retrievers
    for scope in scopes:
        scoped_rows = records if scope == "pooled" else [row for row in records if row["retriever"] == scope]
        if not scoped_rows:
            raise ValueError(f"scope {scope} 没有记录")
        grouped = {label: [row for row in scoped_rows if row["pca_tier"] == label] for label in TIER_LABELS}
        if any(len(rows) == 0 for rows in grouped.values()):
            raise ValueError(f"scope {scope} PCA 三分组出现空组")

        metric_tests = {}
        for metric in METRIC_COLUMNS:
            samples = [[float(row[metric]) for row in grouped[label]] for label in TIER_LABELS]
            stat, p_value = kruskal(*samples)
            metric_tests[metric] = {
                "kruskal_statistic": float(stat),
                "kruskal_p_value": float(p_value),
                "means": {label: float(np.mean(samples[idx])) for idx, label in enumerate(TIER_LABELS)},
            }
        output[scope] = {
            "groups": {label: metric_group_summary(grouped[label]) for label in TIER_LABELS},
            "metric_tests": metric_tests,
        }
    return output


def build_continuous_analysis(records: list[dict]) -> dict:
    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError("连续模型输入为空")

    terms = ["shared_pca_score"]
    omitted_terms = []
    for term in ("query_length", "avg_idf"):
        if int(df[term].nunique()) > 1:
            terms.append(term)
        else:
            omitted_terms.append({"term": term, "reason": "constant"})

    if int(df["retriever"].nunique()) > 1:
        terms.append("C(retriever)")
    else:
        omitted_terms.append({"term": "C(retriever)", "reason": "constant"})

    if int(df["domain"].nunique()) > 1:
        terms.append("C(domain)")
        domain_control_note = "domain_fixed_effect_included"
    else:
        omitted_terms.append({"term": "C(domain)", "reason": "single_domain_constant"})
        domain_control_note = "single_domain_constant_omitted_from_formula"

    control_formula = " + ".join(terms)

    result = {
        "domain_control_note": domain_control_note,
        "omitted_terms": omitted_terms,
        "correlations": {},
        "ols_models": {},
        "glm_binomial_models": {},
    }

    for metric in METRIC_COLUMNS:
        pearson = pearsonr(df["shared_pca_score"], df[metric])
        spearman = spearmanr(df["shared_pca_score"], df[metric])
        if np.isnan(pearson.statistic) or np.isnan(spearman.statistic):
            raise ValueError(f"{metric} 与 shared_pca_score 的相关性为 NaN")
        result["correlations"][metric] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p_value": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_value": float(spearman.pvalue),
        }

        formula = f"{metric} ~ {control_formula}"
        ols_model = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
        result["ols_models"][metric] = {
            "formula": formula,
            "nobs": int(ols_model.nobs),
            "r_squared": float(ols_model.rsquared),
            "shared_pca_score_coef": float(ols_model.params["shared_pca_score"]),
            "shared_pca_score_p_value": float(ols_model.pvalues["shared_pca_score"]),
        }
        if "query_length" in ols_model.params:
            result["ols_models"][metric]["query_length_coef"] = float(ols_model.params["query_length"])
            result["ols_models"][metric]["query_length_p_value"] = float(ols_model.pvalues["query_length"])
        if "avg_idf" in ols_model.params:
            result["ols_models"][metric]["avg_idf_coef"] = float(ols_model.params["avg_idf"])
            result["ols_models"][metric]["avg_idf_p_value"] = float(ols_model.pvalues["avg_idf"])

    glm_formula = f"hit_at10 ~ {control_formula}"
    glm_model = smf.glm(formula=glm_formula, data=df, family=sm.families.Binomial()).fit(cov_type="HC3")
    result["glm_binomial_models"]["hit_at10"] = {
        "formula": glm_formula,
        "nobs": int(glm_model.nobs),
        "shared_pca_score_coef": float(glm_model.params["shared_pca_score"]),
        "shared_pca_score_p_value": float(glm_model.pvalues["shared_pca_score"]),
        "shared_pca_score_odds_ratio": float(np.exp(glm_model.params["shared_pca_score"])),
    }
    if "query_length" in glm_model.params:
        result["glm_binomial_models"]["hit_at10"]["query_length_coef"] = float(glm_model.params["query_length"])
        result["glm_binomial_models"]["hit_at10"]["query_length_p_value"] = float(glm_model.pvalues["query_length"])
    if "avg_idf" in glm_model.params:
        result["glm_binomial_models"]["hit_at10"]["avg_idf_coef"] = float(glm_model.params["avg_idf"])
        result["glm_binomial_models"]["hit_at10"]["avg_idf_p_value"] = float(glm_model.pvalues["avg_idf"])

    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    log("开始读取 query shared PCA")
    query_index = load_query_pca_index()
    log(f"query PCA 记录数: {len(query_index)}")

    log("开始对齐 retrieval 指标")
    records = load_aligned_records(query_index)
    log(f"对齐记录数: {len(records)}")

    log("开始划分 PCA 三分组")
    boundaries = assign_pca_tiers(records)

    log("开始连续模型分析")
    continuous = build_continuous_analysis(records)

    log("开始分组指标分析")
    tier_analysis = build_tier_analysis(records)

    summary = {
        "category": CATEGORY,
        "query_pca_file": str(QUERY_PCA_FILE),
        "retrieval_file": str(RETRIEVAL_FILE),
        "record_file": str(RECORD_FILE),
        "num_records": len(records),
        "num_unique_queries": len({query_key(row["user_id"], row["asin"]) for row in records}),
        "retrievers": sorted({row["retriever"] for row in records}),
        "pca_tier_boundaries": boundaries,
        "continuous_analysis": continuous,
        "tier_analysis": tier_analysis,
    }

    write_jsonl(RECORD_FILE, records)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pooled_hit = tier_analysis["pooled"]["metric_tests"]["hit_at10"]["means"]
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "num_records": len(records),
        "num_unique_queries": summary["num_unique_queries"],
        "hit_at10_pooled_means": pooled_hit,
        "hit_at10_ols_pca_coef": continuous["ols_models"]["hit_at10"]["shared_pca_score_coef"],
        "hit_at10_ols_pca_p": continuous["ols_models"]["hit_at10"]["shared_pca_score_p_value"],
        "hit_at10_glm_pca_odds_ratio": continuous["glm_binomial_models"]["hit_at10"]["shared_pca_score_odds_ratio"],
        "hit_at10_glm_pca_p": continuous["glm_binomial_models"]["hit_at10"]["shared_pca_score_p_value"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
