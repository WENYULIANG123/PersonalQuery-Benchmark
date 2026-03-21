#!/usr/bin/env python3
import os
import sys
import glob
import shutil
import subprocess
import time
from datetime import datetime

INPUT_DIR = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
OUTPUT_DIR = "/home/wlia0047/ar57/wenyu/result/personal_query/01_preference_extraction"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_SCRIPT = os.path.join(SCRIPT_DIR, "01_extract_preferences.py")
MAX_WORKERS = 5


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def clean_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return

    for f in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(path):
            os.remove(path)
            log(f"  🗑️  已删除: {f}")

    log(f"✅ 输出目录已清空: {OUTPUT_DIR}")


def get_user_files():
    pattern = os.path.join(INPUT_DIR, "reviews_*.json")
    files = sorted(glob.glob(pattern))
    return files


def extract_user_id(filepath):
    basename = os.path.basename(filepath)
    return basename.replace("reviews_", "").replace(".json", "")


def run_extraction(input_file, user_id, idx, total):
    log(f"[{idx}/{total}] 🚀 开始处理用户: {user_id}")
    log(f"  输入: {input_file}")

    cmd = [
        sys.executable, "-u", EXTRACT_SCRIPT,
        "--input-file", input_file,
        "--output-dir", OUTPUT_DIR,
        "--max-workers", str(MAX_WORKERS),
    ]

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    output_file = os.path.join(OUTPUT_DIR, f"preferences_{user_id}.json")
    if result.returncode == 0 and os.path.exists(output_file):
        log(f"[{idx}/{total}] ✅ 用户 {user_id} 完成 ({elapsed:.0f}s)")
        return True
    else:
        log(f"[{idx}/{total}] ❌ 用户 {user_id} 失败 (exit={result.returncode}, {elapsed:.0f}s)")
        return False


def main():
    log("=" * 80)
    log("Stage 1: 批量偏好提取 - 所有用户")
    log("=" * 80)

    user_files = get_user_files()
    log(f"发现 {len(user_files)} 个用户文件")
    log(f"输入目录: {INPUT_DIR}")
    log(f"输出目录: {OUTPUT_DIR}")
    log("")

    log("🗑️  清理旧结果...")
    clean_output_dir()
    log("")

    success = 0
    failed = []
    total = len(user_files)

    for idx, input_file in enumerate(user_files, 1):
        user_id = extract_user_id(input_file)
        ok = run_extraction(input_file, user_id, idx, total)
        if ok:
            success += 1
        else:
            failed.append(user_id)
        log("")

    log("=" * 80)
    log("🎯 批量处理完成")
    log("=" * 80)
    log(f"总用户数: {total}")
    log(f"成功: {success}")
    log(f"失败: {len(failed)}")
    if failed:
        log(f"失败用户: {', '.join(failed)}")
    log("=" * 80)


if __name__ == "__main__":
    main()
