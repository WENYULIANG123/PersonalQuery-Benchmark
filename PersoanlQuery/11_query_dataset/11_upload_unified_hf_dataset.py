#!/usr/bin/env python3
"""Prepare and upload a unified Hugging Face dataset repository with three configs."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["HF_HUB_DISABLE_XET"] = "1"

import requests
from huggingface_hub import HfApi


SOURCE_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/11_query_dataset")
UPLOAD_ROOT = Path("/home/wlia0047/ar57/wenyu/result/personal_query/12_personalized_query_hf")
REPO_NAME = "personalized-query"
HF_SSH_TARGET = "m3-login2"

_NETWORK_READY = False
_NETWORK_LOCK = threading.Lock()
_TUNNEL_PROCESS = None
_TUNNEL_PORT = None
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_ORIGINAL_SOCKET_CLASS = socket.socket
_REMOTE_DNS_CACHE: dict[tuple[str, int | str], list] = {}


@dataclass(frozen=True)
class UnifiedConfig:
    config_name: str
    category: str
    title: str


CONFIGS = (
    UnifiedConfig("baby", "Baby_Products", "Baby Products"),
    UnifiedConfig("grocery", "Grocery_and_Gourmet_Food", "Grocery and Gourmet Food"),
    UnifiedConfig("pets", "Pet_Supplies", "Pet Supplies"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload one unified Hugging Face dataset repository.")
    parser.add_argument("--namespace", required=True, help="Hugging Face user or organization namespace.")
    parser.add_argument("--create-repo", action="store_true", help="Create the public dataset repo before upload.")
    parser.add_argument("--max-upload-attempts", type=int, default=5, help="Maximum upload attempts for transient network errors.")
    parser.add_argument(
        "--network-mode",
        choices=["ssh-socks", "direct"],
        default="ssh-socks",
        help="Network mode. ssh-socks routes Hugging Face traffic through the login node.",
    )
    return parser.parse_args()


def get_raw_socket_class():
    if getattr(socket.socket, "__module__", "") == "socks":
        import socks

        return socks._orgsocket
    return _ORIGINAL_SOCKET_CLASS


def choose_local_socks_port() -> int:
    raw_socket_class = get_raw_socket_class()
    with raw_socket_class(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def wait_for_socks_tunnel(process, port: int, timeout_seconds: int = 10) -> None:
    raw_socket_class = get_raw_socket_class()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            stderr = process.stderr.read().strip() if process.stderr else ""
            raise RuntimeError(f"SSH SOCKS tunnel failed: return_code={return_code}, stderr={stderr}")

        with raw_socket_class(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return

        time.sleep(0.1)

    process.terminate()
    raise TimeoutError(f"SSH SOCKS tunnel startup timed out: 127.0.0.1:{port}")


def cleanup_socks_tunnel() -> None:
    global _TUNNEL_PROCESS
    if _TUNNEL_PROCESS is None:
        return
    if _TUNNEL_PROCESS.poll() is None:
        _TUNNEL_PROCESS.terminate()
        try:
            _TUNNEL_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _TUNNEL_PROCESS.kill()
            _TUNNEL_PROCESS.wait(timeout=5)


def should_use_remote_dns(host: str) -> bool:
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False
    if host.endswith(".local"):
        return False
    return any(
        marker in host
        for marker in (
            "huggingface.co",
            "hf.co",
            "hf.space",
            "xethub.hf.co",
            "amazonaws.com",
            "cloudfront.net",
        )
    )


def resolve_host_on_login_node(host: str, port: int | str, family=0, type=0, proto=0, flags=0):
    cache_key = (host, port)
    cached = _REMOTE_DNS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cmd = [
        "ssh",
        "-q",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        HF_SSH_TARGET,
        "getent",
        "ahostsv4",
        host,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
    if result.returncode != 0:
        raise socket.gaierror(f"login-node DNS failed for {host}: {result.stderr.strip()}")

    ips = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        ip = parts[0]
        if ip not in ips:
            ips.append(ip)

    if not ips:
        raise socket.gaierror(f"login-node DNS returned no IPv4 addresses for {host}")

    resolved = []
    for ip in ips:
        resolved.extend(_ORIGINAL_GETADDRINFO(ip, port, family, type, proto, flags))
    _REMOTE_DNS_CACHE[cache_key] = resolved
    print(f"[NETWORK] login-node DNS host={host} ips={ips[:4]}", flush=True)
    return resolved


def patch_dns_for_socks() -> None:
    def patched_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        lookup_host = host.decode("ascii") if isinstance(host, bytes) else host
        if isinstance(lookup_host, str) and should_use_remote_dns(lookup_host):
            return resolve_host_on_login_node(lookup_host, port, family, type, proto, flags)
        return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)

    socket.getaddrinfo = patched_getaddrinfo


def ensure_ssh_socks_network() -> None:
    global _NETWORK_READY, _TUNNEL_PROCESS, _TUNNEL_PORT
    if _NETWORK_READY:
        return

    with _NETWORK_LOCK:
        if _NETWORK_READY:
            return

        import socks

        port = choose_local_socks_port()
        cmd = [
            "ssh",
            "-q",
            "-N",
            "-D",
            f"127.0.0.1:{port}",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            HF_SSH_TARGET,
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for_socks_tunnel(process, port)

        socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", port, rdns=True)
        socket.socket = socks.socksocket
        patch_dns_for_socks()

        _TUNNEL_PROCESS = process
        _TUNNEL_PORT = port
        _NETWORK_READY = True
        atexit.register(cleanup_socks_tunnel)
        print(f"[NETWORK] SSH SOCKS enabled socks=127.0.0.1:{port} ssh_target={HF_SSH_TARGET}", flush=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return data


def required_upload_files() -> set[Path]:
    files = {Path("README.md"), Path("summary.json")}
    for config in CONFIGS:
        files.add(Path(config.category) / "data.jsonl")
        files.add(Path(config.category) / "summary.json")
    return files


def assert_upload_root_clean(expected_files: set[Path]) -> None:
    if not UPLOAD_ROOT.exists():
        return
    if not UPLOAD_ROOT.is_dir():
        raise NotADirectoryError(f"Upload root exists but is not a directory: {UPLOAD_ROOT}")
    existing_files = {path.relative_to(UPLOAD_ROOT) for path in UPLOAD_ROOT.rglob("*") if path.is_file()}
    unexpected_files = sorted(existing_files - expected_files)
    if unexpected_files:
        raise RuntimeError(f"Upload root contains unexpected files: {unexpected_files}")


def copy_required_files() -> None:
    expected_files = required_upload_files()
    assert_upload_root_clean(expected_files)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    shutil.copy2(SOURCE_ROOT / "summary.json", UPLOAD_ROOT / "summary.json")
    for config in CONFIGS:
        source_dir = SOURCE_ROOT / config.category
        target_dir = UPLOAD_ROOT / config.category
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("data.jsonl", "summary.json"):
            source_file = source_dir / filename
            if not source_file.is_file():
                raise FileNotFoundError(f"Required source file does not exist: {source_file}")
            if source_file.stat().st_size == 0:
                raise ValueError(f"Required source file is empty: {source_file}")
            shutil.copy2(source_file, target_dir / filename)


def format_dict_table(values: dict[str, Any], key_title: str, value_title: str) -> str:
    if not values:
        raise ValueError("Cannot format an empty table")
    lines = [f"| {key_title} | {value_title} |", "|---|---:|"]
    for key in sorted(values):
        lines.append(f"| `{key}` | {values[key]} |")
    return "\n".join(lines)


def build_size_table(category_summaries: list[dict[str, Any]]) -> str:
    lines = [
        "| Config | Category | Full Rows |",
        "|---|---|---:|",
    ]
    for config, summary in zip(CONFIGS, category_summaries, strict=True):
        lines.append(f"| `{config.config_name}` | {config.title} | {summary['num_dataset_rows']} |")
    return "\n".join(lines)


def build_card(namespace: str, aggregate_summary: dict[str, Any]) -> str:
    category_summaries = aggregate_summary["category_summaries"]
    if len(category_summaries) != len(CONFIGS):
        raise ValueError("Aggregate summary category count does not match configured categories")

    detail_sections = []
    for config, summary in zip(CONFIGS, category_summaries, strict=True):
        if summary["category"] != config.category:
            raise ValueError(f"Summary category mismatch: {summary['category']} != {config.category}")
        detail_sections.append(
            f"""### `{config.config_name}`: {config.title}

Rows by cluster label:

{format_dict_table(summary["rows_by_cluster_label"], "cluster_label", "rows")}

Rows by cluster index:

{format_dict_table(summary["rows_by_cluster_index"], "cluster_index", "rows")}
"""
        )

    return f"""---
language:
- en
license: other
pretty_name: "Personalized Query"
task_categories:
- text-generation
- text-retrieval
tags:
- personalized-search
- personalized-query
- query-generation
- clustered-query
- synthetic-data
- amazon-reviews
configs:
- config_name: baby
  data_files:
  - split: train
    path: Baby_Products/data.jsonl
- config_name: grocery
  data_files:
  - split: train
    path: Grocery_and_Gourmet_Food/data.jsonl
- config_name: pets
  data_files:
  - split: train
    path: Pet_Supplies/data.jsonl
---

# Personalized Query

This repository contains three personalized product-search query datasets in one Hugging Face dataset page.

Each config corresponds to one product category:

- `baby`: Baby Products
- `grocery`: Grocery and Gourmet Food
- `pets`: Pet Supplies

Each config contains one clean clustered query split.

## Dataset Size

{build_size_table(category_summaries)}

Total rows across all configs: {aggregate_summary["num_dataset_rows"]}

## Category Details

{chr(10).join(detail_sections)}

## Schema

| Field | Type | Description |
|---|---|---|
| `category` | string | Product category. |
| `uuid` | string | User identifier. |
| `asin` | string | Amazon product identifier. |
| `cluster_label` | string | Query-cluster label such as `cluster_0`. |
| `cluster_index` | integer | Query-cluster index aligned with the cluster label. |
| `correct_query` | string | Clean query text used for evaluation. |
| `attrs_used` | object | Product attributes used to generate the query. |

## Loading

```python
from datasets import load_dataset

baby = load_dataset("{namespace}/{REPO_NAME}", name="baby", split="train")
grocery = load_dataset("{namespace}/{REPO_NAME}", name="grocery", split="train")
pets = load_dataset("{namespace}/{REPO_NAME}", name="pets", split="train")
```

## Source Pipeline

- Stage 06 provides the latest clean syntax-depth query set.
- Stage 12 provides the `strict5550_query_gmm_user_profiles.jsonl` cluster assignment for each user-product query.

## Intended Use

This dataset is intended for research on personalized product search, clustered query style analysis, query generation, and retrieval evaluation.

## Data Notes

The queries are synthetic outputs generated from user/product signals in the local Personal Query pipeline. Review license and redistribution requirements for any upstream source data before external reuse.
"""


def write_readme(namespace: str) -> None:
    aggregate_summary = read_json(SOURCE_ROOT / "summary.json")
    readme = build_card(namespace, aggregate_summary)
    (UPLOAD_ROOT / "README.md").write_text(readme, encoding="utf-8")


def verify_upload_root() -> None:
    expected_files = required_upload_files()
    actual_files = {path.relative_to(UPLOAD_ROOT) for path in UPLOAD_ROOT.rglob("*") if path.is_file()}
    missing_files = sorted(expected_files - actual_files)
    unexpected_files = sorted(actual_files - expected_files)
    if missing_files:
        raise RuntimeError(f"Upload root is missing required files: {missing_files}")
    if unexpected_files:
        raise RuntimeError(f"Upload root contains unexpected files: {unexpected_files}")


def upload(namespace: str, create_repo: bool, max_upload_attempts: int) -> str:
    repo_id = f"{namespace}/{REPO_NAME}"
    api = HfApi()
    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    attempts = 0
    while True:
        attempts += 1
        try:
            commit_info = api.upload_folder(
                folder_path=str(UPLOAD_ROOT),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Upload unified personalized clustered query dataset",
            )
            return commit_info.oid
        except (requests.RequestException, OSError) as exc:
            if attempts >= max_upload_attempts:
                raise RuntimeError(f"Upload failed after {attempts} attempts: {exc}") from exc
            print(f"[UPLOAD] retry attempt={attempts} reason={exc}", flush=True)
            time.sleep(2)


def main() -> None:
    args = parse_args()
    if args.network_mode == "ssh-socks":
        ensure_ssh_socks_network()
    elif args.network_mode != "direct":
        raise ValueError(f"Unsupported network mode: {args.network_mode}")

    copy_required_files()
    write_readme(args.namespace)
    verify_upload_root()
    commit_oid = upload(args.namespace, args.create_repo, args.max_upload_attempts)
    print(f"[SUMMARY] repo_id={args.namespace}/{REPO_NAME} commit={commit_oid}", flush=True)


if __name__ == "__main__":
    main()
