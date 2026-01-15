#!/usr/bin/env python3
"""
Cursor Hook: Agent停止跟踪器
跟踪agent会话的停止事件并提供统计
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# 统计文件路径
STATS_FILE = Path(".cursor/agent_stats.log")


def read_stdin() -> Dict[str, Any]:
    """从标准输入读取 JSON 数据"""
    try:
        return json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON input: {e}", file=sys.stderr)
        return {}


def write_stats(timestamp: str, status: str, loop_count: int):
    """写入统计信息到文件"""
    stats_line = f"{timestamp},{status},{loop_count}\n"
    try:
        STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATS_FILE, "a", encoding="utf-8") as f:
            f.write(stats_line)
    except Exception as e:
        print(f"Error writing stats: {e}", file=sys.stderr)


def calculate_stats() -> Dict[str, int]:
    """计算会话统计"""
    stats = {
        "total_sessions": 0,
        "error_count": 0,
        "completed_count": 0
    }
    
    if not STATS_FILE.exists():
        return stats
    
    try:
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
            stats["total_sessions"] = len(lines)
            stats["error_count"] = sum(1 for line in lines if ",error," in line)
            stats["completed_count"] = sum(1 for line in lines if ",completed," in line)
    except Exception as e:
        print(f"Error reading stats: {e}", file=sys.stderr)
    
    return stats


def main():
    """主函数"""
    input_data = read_stdin()
    
    # 提取状态信息
    status = input_data.get("status", "unknown")
    loop_count = input_data.get("loop_count", 0)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 输出调试信息到 stderr
    print("=== AGENT STOP TRACKER ===", file=sys.stderr)
    print(f"Time: {timestamp}", file=sys.stderr)
    print(f"Status: {status}", file=sys.stderr)
    print(f"Loop count: {loop_count}", file=sys.stderr)
    
    # 写入统计信息
    write_stats(timestamp, status, loop_count)
    
    # 计算并显示会话统计
    stats = calculate_stats()
    print("Session statistics:", file=sys.stderr)
    print(f"  Total sessions: {stats['total_sessions']}", file=sys.stderr)
    print(f"  Completed: {stats['completed_count']}", file=sys.stderr)
    print(f"  Errors: {stats['error_count']}", file=sys.stderr)
    
    # 如果是错误状态且未达到最大重试次数，可以触发重试
    if status == "error" and loop_count < 3:
        print("Error detected, suggesting retry...", file=sys.stderr)
        response = {
            "followup_message": "The previous operation failed. Let me try again with a different approach."
        }
        print(json.dumps(response))
    else:
        # 返回空对象表示正常结束
        print(json.dumps({}))


if __name__ == "__main__":
    main()
