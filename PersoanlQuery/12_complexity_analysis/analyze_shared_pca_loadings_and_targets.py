#!/usr/bin/env python3
"""Analyze shared PCA loadings and complexity-target monotonicity for Baby_Products."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Baby_Products"
RESULT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
REVIEW_SENTENCE_FILE = RESULT_DIR / "review_sentence_pca_distribution_sentences.jsonl"
SYNTAX_DEPTH_QUERY_FEATURE_FILE = RESULT_DIR / "single_query_clause_features.jsonl"
LEVEL_QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / "query.json"
SUMMARY_FILE = RESULT_DIR / "shared_pca_loading_and_target_analysis_summary.json"
LEVEL_RECORD_FILE = RESULT_DIR / "shared_pca_level_query_records.jsonl"

EXTRACT_SCRIPT = REPO_ROOT / "PersoanlQuery" / "12_complexity_analysis" / "extract_clause_features_single_query.py"


def log(message: str) -> None:
    print(message, flush=True)


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} 为空")
    return rows


def load_json(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"{path} 为空")
    return payload


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


def feature_names_from_rows(rows: list[dict]) -> list[str]:
    names = list(rows[0]["features"].keys())
    if not names:
        raise ValueError("特征名为空")
    return names


def build_feature_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = np.array(
        [[float(row["features"][name]) for name in feature_names] for row in rows],
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("特征矩阵为空")
    return matrix


def fit_shared_pca(review_rows: list[dict], feature_names: list[str]) -> dict:
    review_matrix = build_feature_matrix(review_rows, feature_names)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(review_matrix)
    pca = PCA(n_components=1, random_state=42)
    raw_scores = pca.fit_transform(standardized).ravel()

    anchor = np.array([float(row["features"]["max_dependency_depth"]) for row in review_rows], dtype=float)
    anchor_corr = np.corrcoef(raw_scores, anchor)[0, 1]
    if np.isnan(anchor_corr):
        raise ValueError("PC1 与 max_dependency_depth 的相关性为 NaN")
    sign = 1.0 if anchor_corr >= 0 else -1.0
    scores = sign * raw_scores
    signed_weights = sign * pca.components_[0]

    feature_correlations = []
    for feature_idx, feature_name in enumerate(feature_names):
        values = review_matrix[:, feature_idx]
        corr = np.corrcoef(scores, values)[0, 1]
        if np.isnan(corr):
            raise ValueError(f"shared_pca_score 与特征 {feature_name} 的相关性为 NaN")
        feature_correlations.append({
            "feature": feature_name,
            "signed_weight": float(signed_weights[feature_idx]),
            "review_score_correlation": float(corr),
        })

    feature_correlations.sort(key=lambda item: abs(item["signed_weight"]), reverse=True)
    return {
        "scaler": scaler,
        "pca": pca,
        "sign": sign,
        "review_scores": scores,
        "anchor_corr": float(sign * anchor_corr),
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "feature_loadings": feature_correlations,
    }


def project_rows(rows: list[dict], feature_names: list[str], scaler: StandardScaler, pca: PCA, sign: float) -> np.ndarray:
    matrix = build_feature_matrix(rows, feature_names)
    return sign * pca.transform(scaler.transform(matrix)).ravel()


def monotonic_summary_from_rows(rows: list[dict], group_key: str, score_key: str) -> dict:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        raw_group = row.get(group_key)
        if raw_group is None:
            raise ValueError(f"记录缺少分组键 {group_key}")
        group = int(raw_group)
        grouped.setdefault(group, []).append(float(row[score_key]))

    ordered_levels = sorted(grouped.keys())
    if len(ordered_levels) < 2:
        raise ValueError(f"{group_key} 的组数不足")

    group_summaries = {}
    means = []
    level_vector = []
    score_vector = []
    for level in ordered_levels:
        values = np.array(grouped[level], dtype=float)
        group_summaries[str(level)] = summarize_array(values)
        means.append(float(np.mean(values)))
        level_vector.extend([level] * len(values))
        score_vector.extend(values.tolist())

    strict_increasing = all(means[idx] < means[idx + 1] for idx in range(len(means) - 1))
    nondecreasing = all(means[idx] <= means[idx + 1] for idx in range(len(means) - 1))
    spear = spearmanr(np.array(level_vector, dtype=float), np.array(score_vector, dtype=float))
    if np.isnan(spear.statistic) or np.isnan(spear.pvalue):
        raise ValueError(f"{group_key} 与分数组间 Spearman 结果为 NaN")
    kw_stat, kw_p = kruskal(*[grouped[level] for level in ordered_levels])
    return {
        "ordered_levels": ordered_levels,
        "group_summaries": group_summaries,
        "group_means": {str(level): mean for level, mean in zip(ordered_levels, means)},
        "strict_increasing": bool(strict_increasing),
        "nondecreasing": bool(nondecreasing),
        "spearman": {
            "rho": float(spear.statistic),
            "p_value": float(spear.pvalue),
        },
        "kruskal_wallis": {
            "statistic": float(kw_stat),
            "p_value": float(kw_p),
        },
    }


def load_extract_module():
    if not EXTRACT_SCRIPT.exists():
        raise FileNotFoundError(f"extract script not found: {EXTRACT_SCRIPT}")
    spec = importlib.util.spec_from_file_location("stage12_single_query_features", EXTRACT_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load extract script: {EXTRACT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_level_query_rows(extract_module) -> list[dict]:
    payload = load_json(LEVEL_QUERY_FILE)
    if not isinstance(payload, list):
        raise TypeError(f"{LEVEL_QUERY_FILE} 顶层必须是 list")

    raw_rows = []
    queries = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TypeError(f"query.json[{idx}] 必须是对象")
        for family in ("acl_query", "ccomp_query"):
            query_info = item.get(family)
            if not isinstance(query_info, dict):
                raise TypeError(f"query.json[{idx}].{family} 必须是对象")
            query_text = query_info.get("query")
            level = query_info.get("level")
            if not isinstance(query_text, str) or not query_text.strip():
                raise ValueError(f"query.json[{idx}].{family}.query 必须是非空字符串")
            if level is None:
                raise KeyError(f"query.json[{idx}].{family} 缺少 level")
            raw_rows.append({
                "user_id": item.get("user_id"),
                "asin": item.get("asin"),
                "query_family": family.replace("_query", ""),
                "target_level": int(level),
                "query": query_text.strip(),
            })
            queries.append(query_text.strip())

    if not raw_rows:
        raise ValueError("query.json 中没有 level query")

    nlp = extract_module.load_spacy_model()
    docs = list(nlp.pipe(queries, batch_size=256))
    if len(docs) != len(raw_rows):
        raise ValueError("query docs 数量与原始记录数量不一致")

    level_rows = []
    for raw_row, doc in zip(raw_rows, docs):
        feature_result = extract_module.extract_clause_features_from_doc(doc, raw_row["query"])
        level_rows.append({
            **raw_row,
            "word_count": feature_result["word_count"],
            "features": feature_result["features"],
        })
    return level_rows


def attach_projected_scores(rows: list[dict], scores: np.ndarray) -> list[dict]:
    if len(rows) != len(scores):
        raise ValueError("记录数与分数数不一致")
    enriched = []
    for row, score in zip(rows, scores):
        new_row = dict(row)
        new_row["shared_pca_score"] = float(score)
        enriched.append(new_row)
    return enriched


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    log("开始读取 review sentence 特征")
    review_rows = load_jsonl(REVIEW_SENTENCE_FILE)
    log(f"review sentence 行数: {len(review_rows)}")

    feature_names = feature_names_from_rows(review_rows)

    log("开始拟合共享 PCA 并导出 loadings")
    fit_result = fit_shared_pca(review_rows, feature_names)

    log("开始读取 syntax-depth query 特征")
    syntax_depth_rows = load_jsonl(SYNTAX_DEPTH_QUERY_FEATURE_FILE)
    syntax_depth_scores = project_rows(
        syntax_depth_rows,
        feature_names,
        fit_result["scaler"],
        fit_result["pca"],
        fit_result["sign"],
    )
    syntax_depth_rows = attach_projected_scores(syntax_depth_rows, syntax_depth_scores)
    syntax_depth_analysis = monotonic_summary_from_rows(syntax_depth_rows, "target_depth", "shared_pca_score")

    syntax_depth_targets = np.array([int(row["target_depth"]) for row in syntax_depth_rows], dtype=float)
    syntax_depth_score_vector = np.array([float(row["shared_pca_score"]) for row in syntax_depth_rows], dtype=float)
    syntax_depth_pearson = pearsonr(syntax_depth_targets, syntax_depth_score_vector)
    if np.isnan(syntax_depth_pearson.statistic) or np.isnan(syntax_depth_pearson.pvalue):
        raise ValueError("target_depth 与 shared_pca_score 的 Pearson 结果为 NaN")

    log("开始读取并解析 level query")
    extract_module = load_extract_module()
    level_rows = build_level_query_rows(extract_module)
    level_feature_names = feature_names_from_rows(level_rows)
    if level_feature_names != feature_names:
        raise ValueError("level query 特征名与共享 PCA 特征名不一致")

    log("开始将 level query 投影到共享 PCA")
    level_scores = project_rows(
        level_rows,
        feature_names,
        fit_result["scaler"],
        fit_result["pca"],
        fit_result["sign"],
    )
    level_rows = attach_projected_scores(level_rows, level_scores)

    family_grouped: dict[str, list[dict]] = {}
    for row in level_rows:
        family_grouped.setdefault(row["query_family"], []).append(row)

    if set(family_grouped.keys()) != {"acl", "ccomp"}:
        raise ValueError(f"level query family 集合异常: {sorted(family_grouped.keys())}")

    level_analysis = {
        "combined": monotonic_summary_from_rows(level_rows, "target_level", "shared_pca_score"),
        "acl": monotonic_summary_from_rows(family_grouped["acl"], "target_level", "shared_pca_score"),
        "ccomp": monotonic_summary_from_rows(family_grouped["ccomp"], "target_level", "shared_pca_score"),
    }

    combined_targets = np.array([int(row["target_level"]) for row in level_rows], dtype=float)
    combined_scores = np.array([float(row["shared_pca_score"]) for row in level_rows], dtype=float)
    combined_pearson = pearsonr(combined_targets, combined_scores)
    if np.isnan(combined_pearson.statistic) or np.isnan(combined_pearson.pvalue):
        raise ValueError("target_level 与 shared_pca_score 的 Pearson 结果为 NaN")

    for family in ("acl", "ccomp"):
        family_targets = np.array([int(row["target_level"]) for row in family_grouped[family]], dtype=float)
        family_scores = np.array([float(row["shared_pca_score"]) for row in family_grouped[family]], dtype=float)
        family_pearson = pearsonr(family_targets, family_scores)
        if np.isnan(family_pearson.statistic) or np.isnan(family_pearson.pvalue):
            raise ValueError(f"{family} target_level 与 shared_pca_score 的 Pearson 结果为 NaN")
        level_analysis[family]["pearson"] = {
            "r": float(family_pearson.statistic),
            "p_value": float(family_pearson.pvalue),
        }

    level_analysis["combined"]["pearson"] = {
        "r": float(combined_pearson.statistic),
        "p_value": float(combined_pearson.pvalue),
    }

    summary = {
        "category": CATEGORY,
        "review_sentence_file": str(REVIEW_SENTENCE_FILE),
        "syntax_depth_query_feature_file": str(SYNTAX_DEPTH_QUERY_FEATURE_FILE),
        "level_query_file": str(LEVEL_QUERY_FILE),
        "feature_names": feature_names,
        "shared_pca": {
            "fit_source": "review_sentences_only",
            "explained_variance_ratio": fit_result["explained_variance_ratio"],
            "anchor_feature": "max_dependency_depth",
            "anchor_correlation": fit_result["anchor_corr"],
            "feature_loadings": fit_result["feature_loadings"],
        },
        "target_depth_analysis": {
            **syntax_depth_analysis,
            "pearson": {
                "r": float(syntax_depth_pearson.statistic),
                "p_value": float(syntax_depth_pearson.pvalue),
            },
        },
        "target_level_analysis": level_analysis,
        "level_record_file": str(LEVEL_RECORD_FILE),
    }

    write_jsonl(LEVEL_RECORD_FILE, level_rows)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log("共享 PCA loadings 与目标复杂度分析完成")
    print(json.dumps({
        "summary_file": str(SUMMARY_FILE),
        "explained_variance_ratio": summary["shared_pca"]["explained_variance_ratio"],
        "anchor_correlation": summary["shared_pca"]["anchor_correlation"],
        "target_depth_strict_increasing": summary["target_depth_analysis"]["strict_increasing"],
        "target_level_acl_strict_increasing": summary["target_level_analysis"]["acl"]["strict_increasing"],
        "target_level_ccomp_strict_increasing": summary["target_level_analysis"]["ccomp"]["strict_increasing"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
