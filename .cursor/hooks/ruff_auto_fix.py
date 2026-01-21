#!/usr/bin/env python3
"""
Cursor Hook: 使用 Ruff 自动修复 Python 文件
在文件编辑后，自动对该文件运行 `ruff --fix`。
"""

import json
import os
import subprocess
import sys
import ast
from collections import deque
from typing import Any, Dict


RUFF_BIN = "/home/wlia0047/ar57/wenyu/ruff-venv/bin/ruff"
TMPDIR = "/home/wlia0047/ar57/wenyu/tmp"

# 仓库根目录（用于解析本地 import -> 文件路径）
REPO_ROOT = "/home/wlia0047/ar57/wenyu"

# 依赖检查的递归深度与最大文件数（避免过慢）
DEFAULT_DEP_DEPTH = 2
DEFAULT_DEP_MAX_FILES = 200


def read_stdin() -> Dict[str, Any]:
    """从标准输入读取 JSON 数据"""
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return {}
    return data


def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _is_within_repo(path: str) -> bool:
    try:
        return os.path.commonpath([os.path.realpath(path), REPO_ROOT]) == REPO_ROOT
    except Exception:
        return False


def _module_to_candidate_paths(module: str, base_dir: str) -> list[str]:
    """
    将模块名转换为可能的文件路径候选（启发式）：
    - <base_dir>/<module_path>.py
    - <base_dir>/<module_path>/__init__.py
    - <repo_root>/<module_path>.py
    - <repo_root>/<module_path>/__init__.py
    """
    if not module:
        return []
    parts = module.split(".")
    rel_py = os.path.join(*parts) + ".py"
    rel_init = os.path.join(*parts, "__init__.py")
    return [
        os.path.join(base_dir, rel_py),
        os.path.join(base_dir, rel_init),
        os.path.join(REPO_ROOT, rel_py),
        os.path.join(REPO_ROOT, rel_init),
    ]


def _resolve_import_from(node: ast.ImportFrom, file_dir: str) -> list[str]:
    """
    解析 `from ... import ...`：
    - 处理相对导入 level（from .foo import bar）
    - 仅返回能映射到 repo 内具体 .py 的候选文件
    """
    module = node.module or ""
    level = int(getattr(node, "level", 0) or 0)

    # 计算相对导入的 base_dir
    base_dir = file_dir
    for _ in range(level):
        base_dir = os.path.dirname(base_dir)

    candidates = []

    # 1) from X import ... -> 先尝试 X 自身
    if module:
        candidates.extend(_module_to_candidate_paths(module, base_dir))

    # 2) 有些项目会用 from pkg import submodule（submodule 是文件）
    #    尝试把 imported name 拼到 module 后面
    for alias in getattr(node, "names", []) or []:
        name = getattr(alias, "name", "") or ""
        if not name or name == "*":
            continue
        if module:
            candidates.extend(_module_to_candidate_paths(f"{module}.{name}", base_dir))
        else:
            candidates.extend(_module_to_candidate_paths(name, base_dir))

    return candidates


def _extract_local_import_deps(entry_file: str, max_depth: int, max_files: int) -> list[str]:
    """
    从 entry_file 出发，静态解析 import，找到 repo 内可能被导入的 .py 文件。
    递归 max_depth 层，最多返回 max_files 个文件。
    """
    entry_file = os.path.realpath(entry_file)
    if not os.path.exists(entry_file) or not entry_file.endswith(".py"):
        return []
    if not _is_within_repo(entry_file):
        return [entry_file]

    seen_files = set()
    out = []

    q = deque([(entry_file, 0)])
    while q and len(out) < max_files:
        path, depth = q.popleft()
        path = os.path.realpath(path)
        if path in seen_files:
            continue
        seen_files.add(path)
        out.append(path)

        if depth >= max_depth:
            continue

        text = _safe_read_text(path)
        if not text:
            continue

        try:
            tree = ast.parse(text, filename=path)
        except SyntaxError:
            # 语法错误本身也要让 ruff 去报；这里无法继续解析依赖
            continue

        file_dir = os.path.dirname(path)
        candidate_paths = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names or []:
                    mod = alias.name or ""
                    candidate_paths.extend(_module_to_candidate_paths(mod, file_dir))
            elif isinstance(node, ast.ImportFrom):
                candidate_paths.extend(_resolve_import_from(node, file_dir))

        for cand in candidate_paths:
            cand = os.path.realpath(cand)
            if (
                cand.endswith(".py")
                and os.path.exists(cand)
                and _is_within_repo(cand)
                and cand not in seen_files
            ):
                q.append((cand, depth + 1))

    return out


def _run_ruff_check(file_paths: list[str], fix: bool) -> subprocess.CompletedProcess:
    """运行 ruff check；fix=True 时会尝试对给定文件集合修复。"""
    cmd = [
        RUFF_BIN,
        "check",
    ]
    if fix:
        cmd += ["--fix", "--unsafe-fixes"]
    cmd += [
        "--output-format",
        "json",
        *file_paths,
    ]
    print(f"[ruff_auto_fix] Running: {' '.join(cmd)}", file=sys.stderr)
    env = os.environ.copy()
    env.setdefault("TMPDIR", TMPDIR)
    return subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def run_ruff_fix(file_path: str) -> None:
    """对指定 Python 文件运行 ruff --fix"""
    if not os.path.exists(file_path):
        print(f"[ruff_auto_fix] File not found: {file_path}", file=sys.stderr)
        return

    if not file_path.endswith(".py"):
        print(f"[ruff_auto_fix] Skip non-Python file: {file_path}", file=sys.stderr)
        return

    try:
        # 1) 先对当前文件做 auto-fix（保持原有行为）
        result = _run_ruff_check([file_path], fix=True)

        # 2) 再对“当前文件 + 可能导入的本地模块”做一次只检查（不改代码）
        dep_depth = int(os.environ.get("RUFF_DEP_CHECK_DEPTH", str(DEFAULT_DEP_DEPTH)) or DEFAULT_DEP_DEPTH)
        dep_max = int(os.environ.get("RUFF_DEP_MAX_FILES", str(DEFAULT_DEP_MAX_FILES)) or DEFAULT_DEP_MAX_FILES)
        dep_files = _extract_local_import_deps(file_path, max_depth=dep_depth, max_files=dep_max)

        # 避免重复打印：如果依赖集合只有当前文件，就不额外跑
        dep_check_result = None
        if len(dep_files) > 1:
            print(
                f"[ruff_auto_fix] 🔎 Dependency check: {len(dep_files)} files "
                f"(depth={dep_depth}, max={dep_max})",
                file=sys.stderr,
            )
            dep_check_result = _run_ruff_check(dep_files, fix=False)

        # 解析 JSON 输出，打印每一条具体违规
        total_violations = 0
        violations_by_code = {}
        
        # helper：解析并汇总一次 ruff 输出
        def _consume_ruff_stdout(stdout_text: str):
            nonlocal total_violations, violations_by_code
            if not stdout_text:
                return
            try:
                data = json.loads(stdout_text)
                
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
            except json.JSONDecodeError:
                # 如果解析失败，就退回到原始 stdout
                print("[ruff_auto_fix] ⚠️  无法解析 Ruff JSON 输出，显示原始输出:", file=sys.stderr)
                print(stdout_text.rstrip("\n"), file=sys.stderr)

        _consume_ruff_stdout(result.stdout)
        if dep_check_result is not None:
            _consume_ruff_stdout(dep_check_result.stdout)

        if result.stderr:
            print("[ruff_auto_fix] Ruff stderr:", file=sys.stderr)
            print(result.stderr.rstrip("\n"), file=sys.stderr)
        if dep_check_result is not None and dep_check_result.stderr:
            print("[ruff_auto_fix] Ruff stderr (dependency check):", file=sys.stderr)
            print(dep_check_result.stderr.rstrip("\n"), file=sys.stderr)

        # 打印汇总信息（汇总包含依赖检查）
        if total_violations == 0:
            print("[ruff_auto_fix] ✅ 全pass - 未发现任何错误", file=sys.stderr)
        else:
            print(f"[ruff_auto_fix] ⚠️  发现 {total_violations} 个违规", file=sys.stderr)
            if violations_by_code:
                code_summary = ", ".join(
                    [f"{code}({count})" for code, count in sorted(violations_by_code.items())]
                )
                print(f"[ruff_auto_fix] 错误类型统计: {code_summary}", file=sys.stderr)

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

