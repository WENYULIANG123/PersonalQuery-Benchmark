#!/usr/bin/env python3
"""
Cursor Hook: 使用 Ruff 自动修复 Python 文件
在文件编辑后，自动对该文件运行 `ruff --fix`。
"""

import json
import os
import subprocess
import sys
from typing import Any, Dict


RUFF_BIN = "/home/wlia0047/ar57/wenyu/ruff-venv/bin/ruff"
TMPDIR = "/home/wlia0047/ar57/wenyu/tmp"


def read_stdin() -> Dict[str, Any]:
    """从标准输入读取 JSON 数据"""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}
    return data


def run_ruff_fix(file_path: str) -> None:
    """对指定 Python 文件运行 ruff --fix"""
    if not os.path.exists(file_path):
        print(f"[ruff_auto_fix] File not found: {file_path}", file=sys.stderr)
        return

    if not file_path.endswith(".py"):
        print(f"[ruff_auto_fix] Skip non-Python file: {file_path}", file=sys.stderr)
        return

    # 确保临时目录可用，避免写到已满的 scratch
    env = os.environ.copy()
    env.setdefault("TMPDIR", TMPDIR)

    # 更强硬：同时启用 unsafe fixes（可能改变代码行为，请谨慎）
    # 使用 JSON 输出，方便解析并打印出每一条具体违规
    cmd = [
        RUFF_BIN,
        "check",
        "--fix",
        "--unsafe-fixes",
        "--output-format",
        "json",
        file_path,
    ]
    print(f"[ruff_auto_fix] Running: {' '.join(cmd)}", file=sys.stderr)

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # 解析 JSON 输出，打印每一条具体违规
        total_violations = 0
        violations_by_code = {}
        
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                
                # 统计违规信息
                for file_result in data:
                    filename = file_result.get("filename", file_path)
                    violations = file_result.get("violations", [])
                    
                    for v in violations:
                        total_violations += 1
                        code = v.get("code", "UNKNOWN")
                        message = v.get("message", "")
                        loc = v.get("location", {})
                        line = loc.get("row", "?")
                        col = loc.get("column", "?")
                        
                        # 统计错误类型
                        if code not in violations_by_code:
                            violations_by_code[code] = 0
                        violations_by_code[code] += 1
                        
                        # 打印详细错误信息
                        print(
                            f"[ruff_auto_fix] ❌ {filename}:{line}:{col} "
                            f"[{code}] {message}",
                            file=sys.stderr,
                        )
                
                # 打印汇总信息
                if total_violations == 0:
                    print("[ruff_auto_fix] ✅ 全pass - 未发现任何错误", file=sys.stderr)
                else:
                    print(f"[ruff_auto_fix] ⚠️  发现 {total_violations} 个违规", file=sys.stderr)
                    if violations_by_code:
                        code_summary = ", ".join([f"{code}({count})" for code, count in sorted(violations_by_code.items())])
                        print(f"[ruff_auto_fix] 错误类型统计: {code_summary}", file=sys.stderr)
                        
            except json.JSONDecodeError:
                # 如果解析失败，就退回到原始 stdout
                print("[ruff_auto_fix] ⚠️  无法解析 Ruff JSON 输出，显示原始输出:", file=sys.stderr)
                print(result.stdout.rstrip("\n"), file=sys.stderr)
        else:
            # 没有 stdout 输出，通常表示没有错误
            print("[ruff_auto_fix] ✅ 全pass - 未发现任何错误", file=sys.stderr)

        if result.stderr:
            print("[ruff_auto_fix] Ruff stderr:", file=sys.stderr)
            print(result.stderr.rstrip("\n"), file=sys.stderr)

        if result.returncode == 0:
            if total_violations == 0:
                print(f"[ruff_auto_fix] ✅ Ruff 检查完成: {file_path} (无错误)", file=sys.stderr)
            else:
                print(f"[ruff_auto_fix] ✅ Ruff 修复完成: {file_path} (已修复 {total_violations} 个问题)", file=sys.stderr)
        else:
            print(f"[ruff_auto_fix] ❌ Ruff 退出码: {result.returncode} (可能有未修复的错误)", file=sys.stderr)
    except FileNotFoundError:
        print(f"[ruff_auto_fix] Ruff binary not found at {RUFF_BIN}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("[ruff_auto_fix] Ruff fix timed out", file=sys.stderr)


def main() -> None:
    data = read_stdin()
    file_path = data.get("file_path") or data.get("path") or ""

    print(f"[ruff_auto_fix] Hook triggered for file: {file_path}", file=sys.stderr)
    if file_path:
        run_ruff_fix(file_path)

    # 按 Cursor hook 协议要求，stdout 必须输出一个 JSON 对象
    print(json.dumps({}))


if __name__ == "__main__":
    main()

