#!/usr/bin/env python3
"""
Stage 2 Pipeline - 数据处理主脚本

自动运行 Stage 2 流程：
1. 02_split_train_holdout.py - 过滤用户偏好数据

默认自动检测 Stage 1 的所有用户偏好文件并批量处理。
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


def log_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def find_user_preference_files(preferences_dir):
    pattern = "preferences_*.json"
    files = list(Path(preferences_dir).glob(pattern))
    user_ids = []
    for f in files:
        filename = f.name
        user_id = filename.replace("preferences_", "").replace(".json", "")
        user_ids.append((user_id, str(f)))
    return user_ids


def run_command(cmd, description):
    log_with_timestamp(f"运行: {description}")
    log_with_timestamp(f"命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        log_with_timestamp(f"❌ 错误: {description} 失败")
        log_with_timestamp(f"STDERR: {result.stderr}")
        return False
    
    log_with_timestamp(f"✅ 完成: {description}")
    if result.stdout:
        print(result.stdout)
    return True


def run_split_train_holdout(preferences_dir, metadata_file, output_dir, user_id=None, 
                            min_attrs=3, min_other_users=3, seed=42):
    script = os.path.join(os.path.dirname(__file__), "02_split_train_holdout.py")
    
    cmd = [
        "python3", script,
        "--preferences-dir", preferences_dir,
        "--metadata-file", metadata_file,
        "--output-dir", output_dir,
        "--min-attrs", str(min_attrs),
        # "--min-other-users", str(min_other_users),  # 已注释：不再检查其他用户属性数
        "--seed", str(seed)
    ]
    
    if user_id:
        cmd.extend(["--user-id", user_id])
    
    return run_command(cmd, f"Stage 2a: Split train/holdout (user: {user_id or 'all'})")








def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 Pipeline - 自动运行完整的数据处理和画像生成流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 处理所有用户（默认）:
    python run_stage2_pipeline.py

2. 只处理特定用户:
    python run_stage2_pipeline.py --user-id A13OFOB1394G31

3. 跳过数据过滤步骤:
    python run_stage2_pipeline.py --skip-split

4. 自定义参数:
    python run_stage2_pipeline.py --min-attrs 8
        """
    )
    
    parser.add_argument(
        "--preferences-dir",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction",
        help="Stage 1 偏好提取输出目录 (默认: result/personal_query/01_preference_extraction)"
    )
    
    parser.add_argument(
        "--metadata-file",
        default="/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json",
        help="产品元数据文件 (默认: data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="/home/wlia0047/ar57/wenyu/result/personal_query/02_processing",
        help="Stage 2 输出目录 (默认: result/personal_query/02_processing)"
    )
    
    parser.add_argument(
        "--user-id",
        help="只处理指定用户 (默认: 处理所有用户)"
    )
    
    parser.add_argument(
        "--min-attrs",
        type=int,
        default=5,
        help="查询集商品最少属性数 (默认: 5)"
    )
    
    
    parser.add_argument(
        "--min-other-users",
        type=int,
        default=3,
        help="商品需要的最少公开属性数 (默认: 3)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="跳过 Stage 2a (split train/holdout)"
    )
    

    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("Stage 2 Pipeline - 数据处理和画像生成")
    log_with_timestamp("=" * 80)
    
    if not os.path.exists(args.preferences_dir):
        log_with_timestamp(f"❌ 错误: preferences 目录不存在: {args.preferences_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.metadata_file):
        log_with_timestamp(f"❌ 错误: metadata 文件不存在: {args.metadata_file}")
        sys.exit(1)
    
    user_files = find_user_preference_files(args.preferences_dir)
    
    if not user_files:
        log_with_timestamp(f"❌ 错误: 未找到任何 preferences_*.json 文件")
        sys.exit(1)
    
    if args.user_id:
        user_files = [(uid, path) for uid, path in user_files if uid == args.user_id]
        if not user_files:
            log_with_timestamp(f"❌ 错误: 未找到用户 {args.user_id} 的偏好文件")
            sys.exit(1)
    
    log_with_timestamp(f"发现 {len(user_files)} 个用户偏好文件:")
    for uid, path in user_files:
        log_with_timestamp(f"  - {uid}: {path}")
    
    log_with_timestamp("")
    log_with_timestamp("配置参数:")
    log_with_timestamp(f"  - min_attrs: {args.min_attrs}")
    log_with_timestamp(f"  - min_other_users: {args.min_other_users}")
    log_with_timestamp(f"  - seed: {args.seed}")
    log_with_timestamp("")
    
    total_steps = len(user_files) * (
        (0 if args.skip_split else 1)
    )
    current_step = 0
    
    success_count = 0
    failed_users = []
    
    for user_id, preferences_file in user_files:
        log_with_timestamp("")
        log_with_timestamp("=" * 80)
        log_with_timestamp(f"处理用户: {user_id}")
        log_with_timestamp("=" * 80)
        
        user_success = True
        
        if not args.skip_split:
            current_step += 1
            log_with_timestamp(f"[{current_step}/{total_steps}] Stage 2a: Split train/holdout")
            
            if not run_split_train_holdout(
                args.preferences_dir,
                args.metadata_file,
                args.output_dir,
                user_id=user_id,
                min_attrs=args.min_attrs,
                min_other_users=args.min_other_users,
                seed=args.seed
            ):
                user_success = False
                failed_users.append((user_id, "split"))
        

        
        if user_success:
            success_count += 1
            log_with_timestamp(f"✅ 用户 {user_id} 处理完成")
        else:
            log_with_timestamp(f"❌ 用户 {user_id} 处理失败")
    
    log_with_timestamp("")
    log_with_timestamp("=" * 80)
    log_with_timestamp("Pipeline 执行完成")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"总用户数: {len(user_files)}")
    log_with_timestamp(f"成功: {success_count}")
    log_with_timestamp(f"失败: {len(user_files) - success_count}")
    
    if failed_users:
        log_with_timestamp("")
        log_with_timestamp("失败详情:")
        for user_id, stage in failed_users:
            log_with_timestamp(f"  - {user_id}: {stage}")
    
    log_with_timestamp("")
    log_with_timestamp("输出文件位置:")
    log_with_timestamp(f"  - {args.output_dir}/{{user_id}}/query.json")
    
    if success_count == len(user_files):
        log_with_timestamp("")
        log_with_timestamp("🎉 所有用户处理成功!")
        sys.exit(0)
    else:
        log_with_timestamp("")
        log_with_timestamp("⚠️  部分用户处理失败，请检查日志")
        sys.exit(1)


if __name__ == "__main__":
    main()
