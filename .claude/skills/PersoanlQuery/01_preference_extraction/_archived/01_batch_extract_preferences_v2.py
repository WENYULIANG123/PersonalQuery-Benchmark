#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

def log_with_timestamp(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def main():
    selected_users_file = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation/selected_users.json"
    reviews_dir = "/fs04/ar57/wenyu/result/personal_query/00_data_preparation"
    output_dir = "/fs04/ar57/wenyu/result/personal_query/01_preference_extraction"
    script_path = "/home/wlia0047/ar57/wenyu/.claude/skills/PersoanlQuery/01_preference_extraction/01_extract_preferences_v2_with_aspects.py"
    max_workers = 5
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("🚀 Stage 1 批量提取: 所有用户偏好 (v2 方案)")
    log_with_timestamp("=" * 80)
    
    if not os.path.exists(selected_users_file):
        log_with_timestamp(f"❌ 错误: 找不到 {selected_users_file}")
        sys.exit(1)
    
    with open(selected_users_file) as f:
        data = json.load(f)
    
    users = data.get('users', [])
    
    if not users:
        log_with_timestamp("❌ 错误: 未找到用户列表")
        sys.exit(1)
    
    log_with_timestamp(f"📊 总用户数: {len(users)}")
    log_with_timestamp(f"👥 用户列表: {', '.join(users)}")
    log_with_timestamp(f"📍 输入目录: {reviews_dir}")
    log_with_timestamp(f"📤 输出目录: {output_dir}")
    log_with_timestamp(f"⚙️  并发数: {max_workers}")
    log_with_timestamp("")
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_time = time.time()
    success_count = 0
    failed_users = []
    user_results = {}
    
    for idx, user_id in enumerate(users, 1):
        log_with_timestamp("=" * 80)
        log_with_timestamp(f"[{idx}/{len(users)}] 正在处理用户: {user_id}")
        log_with_timestamp("=" * 80)
        
        input_file = os.path.join(reviews_dir, f"reviews_{user_id}.json")
        
        if not os.path.exists(input_file):
            log_with_timestamp(f"❌ 错误: 输入文件不存在 {input_file}")
            failed_users.append(user_id)
            user_results[user_id] = {"status": "failed", "reason": "input_file_not_found"}
            continue
        
        with open(input_file) as f:
            user_data = json.load(f)
        num_products = len(user_data.get('results', []))
        log_with_timestamp(f"📦 产品数: {num_products}")
        
        user_start = time.time()
        
        cmd = [
            sys.executable,
            script_path,
            "--input-file", input_file,
            "--output-dir", output_dir,
            "--max-workers", str(max_workers)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            elapsed = time.time() - user_start
            
            if result.returncode == 0:
                output_file = os.path.join(output_dir, f"preferences_{user_id}_v2.json")
                
                if os.path.exists(output_file):
                    with open(output_file) as f:
                        output_data = json.load(f)
                    
                    num_success = len(output_data.get('results', []))
                    log_with_timestamp(f"✅ 成功 | 耗时: {elapsed:.1f}s | 成功产品: {num_success}/{num_products}")
                    
                    success_count += 1
                    user_results[user_id] = {
                        "status": "success",
                        "products": num_products,
                        "successful": num_success,
                        "elapsed": elapsed,
                        "output_file": output_file
                    }
                else:
                    log_with_timestamp(f"❌ 输出文件未生成")
                    failed_users.append(user_id)
                    user_results[user_id] = {"status": "failed", "reason": "output_not_generated"}
            else:
                log_with_timestamp(f"❌ 失败 (exit code {result.returncode})")
                if result.stderr:
                    log_with_timestamp(f"❌ 错误: {result.stderr[:300]}")
                failed_users.append(user_id)
                user_results[user_id] = {"status": "failed", "reason": "script_error"}
        
        except subprocess.TimeoutExpired:
            log_with_timestamp(f"❌ 超时 (>3600s)")
            failed_users.append(user_id)
            user_results[user_id] = {"status": "failed", "reason": "timeout"}
        except Exception as e:
            log_with_timestamp(f"❌ 异常: {str(e)[:200]}")
            failed_users.append(user_id)
            user_results[user_id] = {"status": "failed", "reason": str(e)[:200]}
        
        log_with_timestamp("")
    
    total_elapsed = time.time() - start_time
    
    log_with_timestamp("=" * 80)
    log_with_timestamp("📊 最终摘要")
    log_with_timestamp("=" * 80)
    log_with_timestamp(f"总用户数: {len(users)}")
    log_with_timestamp(f"成功: {success_count} ✅")
    log_with_timestamp(f"失败: {len(failed_users)} ❌")
    log_with_timestamp(f"成功率: {100 * success_count / len(users):.1f}%")
    log_with_timestamp(f"总耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    log_with_timestamp(f"平均耗时/用户: {total_elapsed/len(users):.1f}s")
    
    if failed_users:
        log_with_timestamp(f"\n失败的用户: {', '.join(failed_users)}")
    
    log_with_timestamp("")
    log_with_timestamp(f"📁 输出目录: {output_dir}")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_users": len(users),
        "successful": success_count,
        "failed": len(failed_users),
        "success_rate": f"{100 * success_count / len(users):.1f}%",
        "total_elapsed_seconds": total_elapsed,
        "average_per_user": total_elapsed / len(users),
        "user_results": user_results
    }
    
    summary_file = os.path.join(output_dir, "batch_v2_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    log_with_timestamp(f"✅ 批处理完成！报告已保存到 {summary_file}")
    log_with_timestamp("=" * 80)
    
    return 0 if len(failed_users) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
