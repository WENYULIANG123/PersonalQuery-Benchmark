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


def load_query_rows(query_file: Path) -> list[dict]:
    rows = load_json(query_file)
    normalized_rows: list[dict] = []
    for row in rows:
        user_id = row.get("user_id")
        asin = row.get("asin")
        query_info = row.get("syntax_depth_query")
        if user_id is None or asin is None or query_info is None:
            raise ValueError(f"{query_file} 存在缺少 user_id / asin / syntax_depth_query 的记录")
        query_text = query_info.get("query")
        if not query_text:
            raise ValueError(f"{query_file} 记录缺少 query: user_id={user_id}, asin={asin}")
        normalized_rows.append(
            {
                "original_row": row,
                "user_id": user_id,
                "asin": asin,
                "query_text": query_text,
                "word_count": int(query_info.get("word_count", len(query_text.split()))),
                "target_depth": query_info.get("target_depth"),
                "user_avg_depth": query_info.get("user_avg_depth"),
            }
        )
    return normalized_rows


def extract_feature_matrix(query_rows: list[dict]) -> tuple[list[str], np.ndarray, list[dict]]:
    log("开始抽取 query 句法特征")
    nlp = load_spacy_model()
    feature_names: list[str] | None = None
    feature_rows: list[dict] = []
    matrix: list[list[float]] = []
    for row in query_rows:
        doc = nlp(row["query_text"])
        extracted = extract_clause_features_from_doc(doc, row["query_text"])
        if feature_names is None:
            feature_names = list(extracted.keys())
        elif list(extracted.keys()) != feature_names:
            raise ValueError("句法特征字段顺序不一致")
        values = [float(extracted[name]) for name in feature_names]
        feature_row = dict(row)
        feature_row["features"] = extracted
        feature_rows.append(feature_row)
        matrix.append(values)
    if feature_names is None:
        raise ValueError("未提取到任何特征")
    return feature_names, np.asarray(matrix, dtype=np.float64), feature_rows


def run_pca_selection(feature_matrix: np.ndarray) -> tuple[np.ndarray, StandardScaler, PCA, dict]:
    log("开始标准化、PCA 降维并选择维度")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    full_pca = PCA(random_state=RANDOM_STATE)
    full_pca.fit(scaled)
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)

    selected_dim = PCA_DIM_MAX
    for dim in range(PCA_DIM_MIN, min(PCA_DIM_MAX, scaled.shape[1]) + 1):
        if cumulative_variance[dim - 1] >= 0.90:
            selected_dim = dim
            break

    pca = PCA(n_components=selected_dim, random_state=RANDOM_STATE)
    embedding = pca.fit_transform(scaled)
    pca_summary = {
        "selected_dim": int(selected_dim),
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return embedding, scaler, pca, pca_summary


def select_best_gmm(embedding: np.ndarray) -> tuple[np.ndarray, GaussianMixture, dict]:
    log("开始 GMM 聚类并自动选 K")
    candidate_rows: list[dict] = []
    for k in GMM_K_RANGE:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = gmm.fit_predict(embedding)
        bic = float(gmm.bic(embedding))
        aic = float(gmm.aic(embedding))
        silhouette = float(silhouette_score(embedding, labels)) if len(set(labels)) > 1 else float("-inf")
        candidate_rows.append(
            {
                "k": int(k),
                "bic": bic,
                "aic": aic,
                "silhouette": silhouette,
                "gmm": gmm,
                "labels": labels,
            }
        )

    candidate_rows.sort(key=lambda row: (row["bic"], -row["silhouette"]))
    best = candidate_rows[0]
    selection_summary = {
        "selected_k": int(best["k"]),
        "criterion": "min_bic_then_max_silhouette",
        "candidates": [
            {
                "k": int(row["k"]),
                "bic": float(row["bic"]),
                "aic": float(row["aic"]),
                "silhouette": float(row["silhouette"]),
            }
            for row in candidate_rows
        ],
    }
    return best["labels"], best["gmm"], selection_summary


def remap_cluster_labels(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[str, int]]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_order = [
        original for original, _count in sorted(zip(unique_labels, counts, strict=True), key=lambda item: (-item[1], item[0]))
    ]
    remap = {int(original): int(new_index) for new_index, original in enumerate(label_order)}
    remapped = np.asarray([remap[int(label)] for label in labels], dtype=np.int64)
    cluster_counts = {f"cluster_{new_index}": int(np.sum(remapped == new_index)) for new_index in range(len(label_order))}
    return remapped, remap, cluster_counts


def build_feature_summaries(feature_names: list[str], feature_matrix: np.ndarray, cluster_labels: np.ndarray) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for cluster_index in sorted(set(cluster_labels.tolist())):
        cluster_mask = cluster_labels == cluster_index
        cluster_matrix = feature_matrix[cluster_mask]
        feature_summary = {}
        for feature_index, feature_name in enumerate(feature_names):
            feature_summary[feature_name] = summarize_array(cluster_matrix[:, feature_index])
        summary[f"cluster_{cluster_index}"] = feature_summary
    return summary


def attach_retrieval_results(category: str, user_rows: list[dict], cluster_labels: np.ndarray) -> None:
    retrieval_summary_file = (
        REPO_ROOT / "result" / "personal_query" / "08_retrieval" / category / "retrieval_syntax_depth_summary.json"
    )
    if not retrieval_summary_file.exists():
        raise FileNotFoundError(f"缺少 retrieval summary: {retrieval_summary_file}")

    retrieval_summary = json.loads(retrieval_summary_file.read_text(encoding="utf-8"))
    retrieval_rows = retrieval_summary.get("retriever_results")
    if retrieval_rows is None:
        retrieval_rows = retrieval_summary.get("retriever_group_results")
    if retrieval_rows is None:
        retrieval_rows = retrieval_summary.get("all_results_combined")
    if retrieval_rows is None:
        raise ValueError(f"{retrieval_summary_file} 缺少 retriever_results / retriever_group_results")

    cluster_by_user = {row["user_id"]: int(cluster_labels[idx]) for idx, row in enumerate(user_rows)}
    grouped_summary = {
        "category": category,
        "method": "gmm_query_syntax_feature_clustering",
        "selected_k": int(len(set(cluster_labels.tolist()))),
        "retriever_group_results": [],
    }

    for retriever_row in retrieval_rows:
        retriever_name = retriever_row.get("retriever")
        records = retriever_row.get("records")
        if records is None:
            records = retriever_row.get("all_query_records")
        if retriever_name is None or records is None:
            raise ValueError(f"{retrieval_summary_file} 检索器结果缺少 retriever 或 records")
        grouped_records: dict[int, list[dict]] = defaultdict(list)
        for record in records:
            user_id = record.get("user_id")
            if user_id is None:
                raise ValueError(f"{retrieval_summary_file} record 缺少 user_id")
            cluster_index = cluster_by_user.get(user_id)
            if cluster_index is None:
                raise KeyError(user_id)
            grouped_records[cluster_index].append(record)

        cluster_hit_at10 = {}
        for cluster_index, cluster_records in grouped_records.items():
            hit_values = [float(record["hit_at10"]) for record in cluster_records]
            cluster_hit_at10[f"cluster_{cluster_index}"] = float(np.mean(hit_values))

        hit_values = np.asarray(list(cluster_hit_at10.values()), dtype=np.float64)
        kruskal_stat, kruskal_pvalue = kruskal(
            *[
                np.asarray([float(record["hit_at10"]) for record in grouped_records[cluster_index]], dtype=np.float64)
                for cluster_index in sorted(grouped_records.keys())
            ]
        )
        grouped_summary["retriever_group_results"].append(
            {
                "retriever": retriever_name,
                "cluster_hit_at10": cluster_hit_at10,
                "hit_at10_gap": float(np.max(hit_values) - np.min(hit_values)),
                "hit_at10_kruskal_statistic": float(kruskal_stat),
                "hit_at10_kruskal_pvalue": float(kruskal_pvalue),
            }
        )

    output_file = (
        REPO_ROOT / "result" / "personal_query" / "08_retrieval" / category / "retrieval_by_strict5550_query_gmm_summary.json"
    )
    output_file.write_text(json.dumps(grouped_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入 retrieval cluster summary: {output_file}")


def run_query_gmm_pipeline(
    category: str = DEFAULT_CATEGORY,
    query_file: Path | None = None,
    write_back_to_query_file: bool = False,
    attach_retrieval: bool = True,
) -> dict:
    clause_dir = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / category
    clause_dir.mkdir(parents=True, exist_ok=True)
    if query_file is None:
        query_file = (
            REPO_ROOT
            / "result"
            / "personal_query"
            / "06_query"
            / category
            / "query_by_syntax_depth_vades_lite_sentence_user_distribution_train10_holdout10.json"
        )
    if not query_file.exists():
        raise FileNotFoundError(f"缺少 query 文件: {query_file}")

    feature_file = clause_dir / "strict5550_query_gmm_features.jsonl"
    user_file = clause_dir / "strict5550_query_gmm_user_profiles.jsonl"
    summary_file = clause_dir / "strict5550_query_gmm_summary.json"

    log(f"开始读取 query 文件: {query_file}")
    query_rows = load_query_rows(query_file)
    feature_names, feature_matrix, feature_rows = extract_feature_matrix(query_rows)
    embedding, _scaler, _pca, pca_summary = run_pca_selection(feature_matrix)
    raw_labels, _gmm, selection_summary = select_best_gmm(embedding)
    cluster_labels, _remap, cluster_counts = remap_cluster_labels(raw_labels)
    cluster_feature_summaries = build_feature_summaries(feature_names, feature_matrix, cluster_labels)

    feature_output_rows = []
    user_output_rows = []
    for idx, row in enumerate(feature_rows):
        cluster_index = int(cluster_labels[idx])
        cluster_label = f"cluster_{cluster_index}"
        feature_output_rows.append(
            {
                "user_id": row["user_id"],
                "asin": row["asin"],
                "cluster_label": cluster_label,
                "cluster_index": cluster_index,
                "query_text": row["query_text"],
                "word_count": row["word_count"],
                "target_depth": row["target_depth"],
                "user_avg_depth": row["user_avg_depth"],
                "features": row["features"],
                "pca_embedding": embedding[idx].tolist(),
            }
        )
        user_output_rows.append(
            {
                "user_id": row["user_id"],
                "asin": row["asin"],
                "cluster_label": cluster_label,
                "cluster_index": cluster_index,
                "query_text": row["query_text"],
                "word_count": row["word_count"],
                "target_depth": row["target_depth"],
                "user_avg_depth": row["user_avg_depth"],
                "pca_embedding": embedding[idx].tolist(),
            }
        )
        if write_back_to_query_file:
            row["original_row"]["query_cluster_label"] = cluster_label
            row["original_row"]["query_cluster_index"] = cluster_index

    write_jsonl(feature_file, feature_output_rows)
    write_jsonl(user_file, user_output_rows)

    if write_back_to_query_file:
        query_file.write_text(
            json.dumps([row["original_row"] for row in query_rows], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = {
        "category": category,
        "method": "gmm_query_syntax_feature_clustering",
        "query_file": str(query_file),
        "feature_file": str(feature_file),
        "user_file": str(user_file),
        "retrieval_summary_file": str(
            REPO_ROOT / "result" / "personal_query" / "08_retrieval" / category / "retrieval_by_strict5550_query_gmm_summary.json"
        ),
        "feature_names": feature_names,
        "pca": pca_summary,
        "cluster_selection": selection_summary,
        "cluster_counts": cluster_counts,
        "cluster_feature_summaries": cluster_feature_summaries,
        "write_back_to_query_file": bool(write_back_to_query_file),
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"已写入 query GMM summary: {summary_file}")

    if attach_retrieval:
        attach_retrieval_results(category, feature_rows, cluster_labels)
    return summary


def main():
    run_query_gmm_pipeline()


if __name__ == "__main__":
    main()
