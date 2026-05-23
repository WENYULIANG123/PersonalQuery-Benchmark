#!/usr/bin/env python3
"""Cluster selected queries by syntactic features with GMM and optionally write labels back to query JSON."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from extract_clause_features_single_query import extract_clause_features_from_doc, load_spacy_model


REPO_ROOT = Path("/fs04/ar57/wenyu")
DEFAULT_CATEGORY = "Baby_Products"
PCA_DIM_MIN = 5
PCA_DIM_MAX = 10
GMM_K_RANGE = [2, 3, 4, 5, 6, 7, 8]
RANDOM_STATE = 42


def log(message: str) -> None:
    print(message, flush=True)


def load_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} 必须是非空列表")
    return data


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


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


def feature_matrix(rows: list[dict], feature_names: list[str]) -> np.ndarray:
    matrix = np.array([[float(row["features"][name]) for name in feature_names] for row in rows], dtype=float)
    if matrix.shape[0] == 0:
        raise ValueError("特征矩阵为空")
    return matrix


def choose_pca_dim(X: np.ndarray) -> tuple[int, dict]:
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    pca = PCA(n_components=min(PCA_DIM_MAX, X_std.shape[1]), random_state=RANDOM_STATE)
    pca.fit(X_std)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    target_dim = None
    for idx, value in enumerate(cumulative, start=1):
        if value >= 0.90 and idx >= PCA_DIM_MIN:
            target_dim = idx
            break
    if target_dim is None:
        target_dim = min(PCA_DIM_MAX, X_std.shape[1])
    return target_dim, {
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_explained_variance_ratio": [float(v) for v in cumulative],
        "selected_pca_dim": int(target_dim),
    }


def choose_gmm(X: np.ndarray) -> tuple[GaussianMixture, np.ndarray, dict]:
    candidates = []
    for k in GMM_K_RANGE:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE, n_init=5)
        labels = gmm.fit_predict(X)
        if len(np.unique(labels)) != k:
            raise ValueError(f"GMM 期望 {k} 个簇，实际得到 {len(np.unique(labels))} 个")
        candidates.append(
            {
                "k": k,
                "gmm": gmm,
                "labels": labels,
                "bic": float(gmm.bic(X)),
                "aic": float(gmm.aic(X)),
                "silhouette": float(silhouette_score(X, labels)),
            }
        )
    best = min(candidates, key=lambda item: (item["bic"], -item["silhouette"]))
    summary = {
        "selected_k": int(best["k"]),
        "selection_rule": "min_bic_then_max_silhouette",
        "candidates": [
            {"k": int(item["k"]), "bic": item["bic"], "aic": item["aic"], "silhouette": item["silhouette"]}
            for item in candidates
        ],
    }
    return best["gmm"], best["labels"], summary


def cluster_name_map(labels: np.ndarray) -> dict[int, str]:
    valid = sorted(set(labels))
    counts = {lab: int(np.sum(labels == lab)) for lab in valid}
    ordered = sorted(valid, key=lambda lab: counts[lab], reverse=True)
    return {lab: f"cluster_{idx}" for idx, lab in enumerate(ordered)}


def build_paths(category: str, query_file: Path | None = None) -> dict[str, Path]:
    clause_dir = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / category
    retrieval_dir = REPO_ROOT / "result" / "personal_query" / "08_retrieval" / category
    default_query_file = (
        REPO_ROOT
        / "result"
        / "personal_query"
        / "06_query"
        / category
        / "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
    )
    selected_query_file = query_file if query_file is not None else default_query_file
    return {
        "query_file": selected_query_file,
        "retrieval_summary_file": retrieval_dir / "retrieval_syntax_depth_summary.json",
        "feature_file": clause_dir / "strict5550_query_gmm_features.jsonl",
        "summary_file": clause_dir / "strict5550_query_gmm_summary.json",
        "user_file": clause_dir / "strict5550_query_gmm_user_profiles.jsonl",
        "retrieval_out_file": retrieval_dir / "retrieval_by_strict5550_query_gmm_summary.json",
    }


def enrich_query_rows_with_clusters(query_rows: list[dict], cluster_labels: list[str], cluster_indices: list[int]) -> list[dict]:
    if len(query_rows) != len(cluster_labels) or len(query_rows) != len(cluster_indices):
        raise ValueError("query rows 与 cluster labels 数量不一致")
    enriched_rows = []
    for row, cluster_label, cluster_index in zip(query_rows, cluster_labels, cluster_indices, strict=True):
        enriched = json.loads(json.dumps(row, ensure_ascii=False))
        enriched["query_cluster_label"] = cluster_label
        enriched["query_cluster_index"] = cluster_index
        enriched_rows.append(enriched)
    return enriched_rows


def run_query_gmm_pipeline(
    category: str,
    query_file: Path | None = None,
    write_back_to_query_file: bool = False,
    attach_retrieval: bool = True,
) -> dict:
    paths = build_paths(category, query_file=query_file)
    query_path = paths["query_file"]
    log(f"开始读取 query 文件: {query_path}")
    query_rows = load_json(query_path)
    nlp = load_spacy_model()

    log("开始抽取 query 句法特征")
    feature_rows = []
    feature_names = None
    for idx, row in enumerate(query_rows, start=1):
        user_id = row["user_id"]
        asin = row["asin"]
        query_info = row["syntax_depth_query"]
        query_text = query_info["query"]
        doc = nlp(query_text)
        extracted = extract_clause_features_from_doc(doc, query_text)
        features = extracted["features"]
        if feature_names is None:
            feature_names = list(features.keys())
        elif list(features.keys()) != feature_names:
            raise ValueError(f"第 {idx} 条 query 特征名不一致")
        feature_rows.append(
            {
                "user_id": user_id,
                "asin": asin,
                "query": query_text,
                "target_depth": int(query_info["target_depth"]),
                "actual_depth": query_info["actual_depth"],
                "user_avg_depth": float(query_info["user_avg_depth"]),
                "word_count": int(extracted["word_count"]),
                "features": features,
            }
        )
    if feature_names is None:
        raise ValueError("没有抽取到 query 特征")

    paths["feature_file"].parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths["feature_file"], feature_rows)

    X_raw = feature_matrix(feature_rows, feature_names)
    log("开始标准化、PCA 降维并选择维度")
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)
    pca_dim, pca_summary = choose_pca_dim(X_raw)
    pca = PCA(n_components=pca_dim, random_state=RANDOM_STATE)
    X = pca.fit_transform(X_std)
    pca_summary["selected_pca_dim"] = int(pca_dim)
    pca_summary["selected_pca_explained_variance_ratio_sum"] = float(np.sum(pca.explained_variance_ratio_))
    pca_summary["selected_pca_explained_variance_ratio"] = [float(v) for v in pca.explained_variance_ratio_]

    log("开始 GMM 聚类并自动选 K")
    _, labels, selection = choose_gmm(X)
    mapping = cluster_name_map(labels)

    cluster_counts = {mapping[lab]: int(np.sum(labels == lab)) for lab in sorted(set(labels))}
    cluster_metric_summary = {}
    for lab in sorted(set(labels)):
        rows_in_cluster = [r for r, lb in zip(feature_rows, labels, strict=True) if lb == lab]
        feature_means = {name: float(np.mean([float(r["features"][name]) for r in rows_in_cluster])) for name in feature_names}
        top_features = sorted(feature_means.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        cluster_metric_summary[mapping[lab]] = {
            "feature_means": feature_means,
            "top_features_by_abs_mean": [{"feature": name, "mean": value} for name, value in top_features],
        }

    cluster_labels = [mapping[int(label)] for label in labels]
    cluster_indices = [int(label.split("_")[1]) for label in cluster_labels]
    user_rows = []
    for row, label, cluster_index, pca_score in zip(feature_rows, cluster_labels, cluster_indices, X, strict=True):
        user_rows.append(
            {
                "user_id": row["user_id"],
                "asin": row["asin"],
                "cluster_label": label,
                "cluster_index": cluster_index,
                "query_text": row["query"],
                "word_count": int(row["word_count"]),
                "target_depth": int(row["target_depth"]),
                "user_avg_depth": float(row["user_avg_depth"]),
                "pca_embedding": [float(v) for v in pca_score],
            }
        )
    write_jsonl(paths["user_file"], user_rows)

    if write_back_to_query_file:
        log(f"开始回写 cluster 标签到 query 文件: {query_path}")
        enriched_query_rows = enrich_query_rows_with_clusters(query_rows, cluster_labels, cluster_indices)
        query_path.write_text(json.dumps(enriched_query_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "category": category,
        "method": "gmm_query_syntax_feature_clustering",
        "query_file": str(query_path),
        "feature_file": str(paths["feature_file"]),
        "user_file": str(paths["user_file"]),
        "retrieval_summary_file": str(paths["retrieval_summary_file"]),
        "feature_names": feature_names,
        "pca": pca_summary,
        "cluster_selection": selection,
        "cluster_counts": cluster_counts,
        "cluster_feature_summaries": cluster_metric_summary,
        "write_back_to_query_file": bool(write_back_to_query_file),
    }
    paths["summary_file"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if attach_retrieval:
        retrieval_summary = json.loads(paths["retrieval_summary_file"].read_text(encoding="utf-8"))
        results_key = "('syntax_depth', 'correct')"
        retriever_results = retrieval_summary["results_by_category_and_type"][results_key]
        metric_names = ["p_at1", "p_at3", "p_at5", "p_at10", "n_at10", "mrr_at10", "hit_at10"]
        group_by_user = {row["user_id"]: row["cluster_label"] for row in user_rows}
        retriever_group_rows = []
        for retriever_result in retriever_results:
            retriever_name = retriever_result["retriever"]
            records = retriever_result.get("all_query_records")
            if not isinstance(records, list) or len(records) == 0:
                raise ValueError(f"retriever {retriever_name} 缺少 all_query_records")
            grouped_records: dict[str, list[dict]] = defaultdict(list)
            for record in records:
                grouped_records[group_by_user[record["user_id"]]].append(record)
            group_names = sorted(grouped_records.keys())
            hit_groups = [np.array([float(row["hit_at10"]) for row in grouped_records[label]], dtype=float) for label in group_names]
            hit_kruskal = kruskal(*hit_groups)
            group_mean_metrics = {
                label: {
                    metric_name: float(np.mean([float(row[metric_name]) for row in grouped_records[label]]))
                    for metric_name in metric_names
                }
                for label in group_names
            }
            retriever_group_rows.append(
                {
                    "retriever": retriever_name,
                    "num_records": len(records),
                    "group_counts": {label: len(grouped_records[label]) for label in group_names},
                    "group_mean_metrics": group_mean_metrics,
                    "hit_at10_gap": float(max(group_mean_metrics[label]["hit_at10"] for label in group_names) - min(group_mean_metrics[label]["hit_at10"] for label in group_names)),
                    "hit_at10_kruskal_pvalue": float(hit_kruskal.pvalue),
                    "hit_at10_kruskal_statistic": float(hit_kruskal.statistic),
                }
            )
        retrieval_out = {
            "category": category,
            "method": "gmm_query_syntax_feature_clustering",
            "query_file": str(query_path),
            "source_retrieval_summary_file": str(paths["retrieval_summary_file"]),
            "selected_k": selection["selected_k"],
            "cluster_counts": cluster_counts,
            "retriever_group_results": sorted(retriever_group_rows, key=lambda row: row["hit_at10_gap"], reverse=True),
        }
        paths["retrieval_out_file"].write_text(json.dumps(retrieval_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        retrieval_out = None

    return {
        "summary_file": str(paths["summary_file"]),
        "feature_file": str(paths["feature_file"]),
        "user_file": str(paths["user_file"]),
        "retrieval_out_file": None if retrieval_out is None else str(paths["retrieval_out_file"]),
        "query_file": str(query_path),
        "selected_k": selection["selected_k"],
        "cluster_counts": cluster_counts,
    }


def main() -> None:
    result = run_query_gmm_pipeline(
        category=DEFAULT_CATEGORY,
        query_file=None,
        write_back_to_query_file=False,
        attach_retrieval=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
