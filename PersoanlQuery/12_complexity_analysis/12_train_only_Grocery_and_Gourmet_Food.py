#!/usr/bin/env python3
"""Run train-only pipeline for Grocery_and_Gourmet_Food (load cached parse results and train model)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/fs04/ar57/wenyu")
CATEGORY = "Grocery_and_Gourmet_Food"

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
import train_vades_lite_sentence_latent_threshold as train_module


def log(message: str) -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> None:
    train_module.REPO_ROOT = REPO_ROOT
    train_module.CATEGORY = CATEGORY
    train_module.INPUT_DIR = REPO_ROOT / "result" / "personal_query" / "12_complexity_analysis_clause_features" / CATEGORY
    train_module.REVIEW_SOURCE_FILE = REPO_ROOT / "result" / "personal_query" / "01_preference_extraction" / CATEGORY / "stage1_filtered_users_reviews.json"
    train_module.RAW_CANDIDATE_QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / "query_by_syntax_depth_no_depth_check_10.json"
    train_module.CANDIDATE_QUERY_FILE = train_module.INPUT_DIR / "query_10_candidates_clause_features_joint_fisher_shared_pca_k3.jsonl"
    train_module.SENTENCE_EXTRACT_CACHE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_extracted_sentences.jsonl"
    train_module.SUMMARY_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_summary.json"
    train_module.DETAIL_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_epoch_details.jsonl"
    train_module.USER_PROFILE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_user_profiles.jsonl"
    train_module.SENTENCE_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_sentences.jsonl"
    train_module.EXCLUDED_USER_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_excluded_users.jsonl"
    train_module.SELECTED_RECORD_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_selected_query_records.jsonl"
    train_module.REJECTED_RECORD_FILE = train_module.INPUT_DIR / f"{train_module.OUTPUT_TAG}_rejected_query_records.jsonl"
    train_module.QUERY_FILE = REPO_ROOT / "result" / "personal_query" / "06_query" / CATEGORY / f"query_by_syntax_depth_{train_module.OUTPUT_TAG}.json"

    log(f"训练阶段开始 - CATEGORY: {CATEGORY}")
    
    # 检查解析结果是否存在
    if not train_module.SENTENCE_FILE.exists():
        log(f"错误: 解析结果不存在，请先运行解析脚本")
        log(f"期望文件: {train_module.SENTENCE_FILE}")
        sys.exit(1)
    
    # Step 1: 加载候选 query
    log("Step 1: 加载候选 query")
    candidate_rows = train_module.load_candidate_query_rows()
    candidate_user_ids = {row["user_id"] for row in candidate_rows}
    log(f"候选用户数: {len(candidate_user_ids)}")
    
    # Step 2: 加载解析结果
    log("Step 2: 加载解析结果")
    with train_module.SENTENCE_FILE.open("r", encoding="utf-8") as f:
        enriched_sentence_rows = [json.loads(line) for line in f if line.strip()]
    log(f"已加载 {len(enriched_sentence_rows)} 条解析句子")
    
    # 获取 feature_names
    if enriched_sentence_rows:
        feature_names = list(enriched_sentence_rows[0]["features"].keys())
    else:
        log("错误: 没有解析句子")
        sys.exit(1)
    
    # Step 3: 加载排除用户
    log("Step 3: 加载排除用户")
    if train_module.EXCLUDED_USER_FILE.exists():
        with train_module.EXCLUDED_USER_FILE.open("r", encoding="utf-8") as f:
            excluded_rows = [json.loads(line) for line in f if line.strip()]
    else:
        excluded_rows = []
    log(f"排除用户数: {len(excluded_rows)}")
    
    # 获取用户 ID 列表
    user_ids = list({row["user_id"] for row in enriched_sentence_rows})
    log(f"用户数: {len(user_ids)}")
    
    # Step 4: 构建训练数据集
    log("Step 4: 构建训练数据集")
    user_ids_filtered, dataset = train_module.build_training_dataset(enriched_sentence_rows, feature_names)
    log(f"训练数据集构建完成: {len(user_ids_filtered)} 用户")
    
    # Step 5: 训练模型
    log("Step 5: 训练 VADES 模型")
    train_module.DEVICE = train_module.infer_device()
    train_module.set_random_seed(train_module.SEED)
    log(f"运行设备: {train_module.DEVICE}")
    encoder, user_table, training_info = train_module.train_vades_user_distribution_model(user_ids_filtered, dataset, feature_names)
    log("模型训练完成")
    
    # Step 6: 推理用户分布
    log("Step 6: 推理用户句子分布")
    sentence_output_rows, user_profile_rows = train_module.infer_user_sentence_distributions(
        encoder, user_table, user_ids_filtered, dataset
    )
    log(f"推理完成: {len(sentence_output_rows)} 句子, {len(user_profile_rows)} 用户")
    
    # Step 7: 校准阈值
    log("Step 7: 校准绝对阈值")
    abs_thresholds = train_module.calibrate_absolute_threshold_with_unseen_holdout(
        encoder, user_table, user_ids_filtered, dataset
    )
    log("阈值校准完成")
    
    # Step 8: 排序和选择 query
    log("Step 8: 排序和选择 query")
    selected_rows, rejected_rows, query_output_rows = train_module.rank_and_select_queries(
        encoder=encoder,
        user_table=user_table,
        dataset=dataset,
        candidate_rows=candidate_rows,
        user_ids=user_ids_filtered,
        user_profile_rows=user_profile_rows,
        abs_thresholds=abs_thresholds,
    )
    log(f"Query 选择完成: {len(selected_rows)} 选中, {len(rejected_rows)} 拒绝")
    
    # Step 9: 保存结果
    log("Step 9: 保存训练结果")
    train_module.write_jsonl(train_module.SENTENCE_FILE, sentence_output_rows)
    train_module.write_jsonl(train_module.USER_PROFILE_FILE, user_profile_rows)
    train_module.write_jsonl(train_module.SELECTED_RECORD_FILE, selected_rows)
    train_module.write_jsonl(train_module.REJECTED_RECORD_FILE, rejected_rows)
    train_module.QUERY_FILE.write_text(json.dumps(query_output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Step 10: 保存摘要
    log("Step 10: 保存训练摘要")
    summary = train_module.build_summary(
        user_ids=user_ids_filtered,
        sentence_rows=sentence_output_rows,
        excluded_rows=excluded_rows,
        feature_names=feature_names,
        training_info=training_info,
        user_profile_rows=user_profile_rows,
        selected_rows=selected_rows,
        rejected_rows=rejected_rows,
        query_output_rows=query_output_rows,
    )
    train_module.SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    train_module.write_jsonl(train_module.DETAIL_FILE, training_info["epochs"])
    
    log("训练阶段完成！")
    log(f"结果文件:")
    log(f"  - 句子: {train_module.SENTENCE_FILE}")
    log(f"  - 用户画像: {train_module.USER_PROFILE_FILE}")
    log(f"  - 选中 query: {train_module.SELECTED_RECORD_FILE}")
    log(f"  - 摘要: {train_module.SUMMARY_FILE}")


if __name__ == "__main__":
    main()
