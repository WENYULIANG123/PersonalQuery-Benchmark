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
        if result.stdout:
            try:
                data = json.loads(result.stdout)
                print("[ruff_auto_fix] Ruff violations (before/after fix):", file=sys.stderr)
                for file_result in data:
                    for v in file_result.get("violations", []):
                        code = v.get("code")
                        message = v.get("message")
                        loc = v.get("location", {})
                        line = loc.get("row")
                        col = loc.get("column")
                        print(
                            f"  {file_result.get('filename')}:{line}:{col} "
                            f"[{code}] {message}",
                            file=sys.stderr,
                        )
            except json.JSONDecodeError:
                # 如果解析失败，就退回到原始 stdout
                print("[ruff_auto_fix] Ruff raw stdout:", file=sys.stderr)
                print(result.stdout.rstrip("\n"), file=sys.stderr)

        if result.stderr:
            print("[ruff_auto_fix] Ruff stderr:", file=sys.stderr)
            print(result.stderr.rstrip("\n"), file=sys.stderr)

        if result.returncode == 0:
            print(f"[ruff_auto_fix] Ruff fix completed for {file_path}", file=sys.stderr)
        else:
            print(f"[ruff_auto_fix] Ruff exited with code {result.returncode}", file=sys.stderr)
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

