#!/usr/bin/env python3
"""Gemma-4-E2B-it 持久化推理服务器

使用 sbatch_wrapper --gpu 提交此脚本，服务器将长期运行
通过 HTTP API 提供推理服务
"""

import subprocess
import time
import os
import signal
import json
import re
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

CONFIG_FILE = "/home/wlia0047/ar57/wenyu/PersoanlQuery/06_query/vllm_config.json"
LOG_FILE = "/home/wlia0047/ar57/wenyu/logs/gemma_server.log"

# 全局模型和处理器
model = None
processor = None
model_loaded = False


def get_slurm_info():
    """获取 SLURM 节点和作业信息"""
    node_name = None
    job_id = None
    try:
        result = subprocess.run(["hostname"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            node_name = result.stdout.strip()
    except:
        pass
    try:
        result = subprocess.run(
            ["bash", "-c", "echo ${SLURM_JOB_ID:-${SLURM_JOBID:-none}}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            job_id = result.stdout.strip()
    except:
        pass
    return node_name, job_id


def update_config(status, node_name=None, job_id=None, base_url=None):
    """更新配置文件"""
    config = {
        "model_name": "google/gemma-4-E2B-it",
        "base_url": base_url or "http://localhost:8000",
        "node_name": node_name,
        "job_id": job_id,
        "status": status,
        "max_model_len": 8192,
        "last_updated": datetime.now().isoformat()
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"配置文件已更新: {CONFIG_FILE}")
    except Exception as e:
        print(f"更新配置文件失败: {e}")


def load_model():
    """加载 Gemma 模型"""
    global model, processor, model_loaded
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    print("正在加载 Gemma-4-E2B-it 模型...")
    MODEL_ID = "google/gemma-4-E2B-it"

    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model_loaded = True
        print("✅ Gemma 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


def generate_text(prompt: str, max_tokens: int = 1024, temperature: float = 0.8) -> str:
    """生成文本"""
    global model, processor

    if model is None or processor is None:
        raise RuntimeError("模型未加载")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0
    )
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    return response


class APIHandler(BaseHTTPRequestHandler):
    """简单的 HTTP API 处理器"""

    def log_message(self, format, *args):
        """自定义日志格式"""
        print(f"[API] {args[0]}")

    def do_POST(self):
        """处理 POST 请求"""
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                prompt = data.get("messages", [])[-1].get("content", "")
                max_tokens = data.get("max_tokens", 1024)
                temperature = data.get("temperature", 0.8)

                response_text = generate_text(prompt, max_tokens, temperature)

                response = {
                    "choices": [{
                        "message": {
                            "content": response_text
                        }
                    }]
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                error = {"error": str(e)}
                self.wfile.write(json.dumps(error).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        """处理 GET 请求（健康检查）"""
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "data": [{
                    "id": "google/gemma-4-E2B-it",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "google"
                }]
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    global model_loaded

    print("=" * 60)
    print("Gemma-4-E2B-it 持久化服务器")
    print("=" * 60)

    node_name, job_id = get_slurm_info()
    print(f"\n节点: {node_name}")
    print(f"作业ID: {job_id}")

    print("\n检查 GPU...")
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout[:500] if result.stdout else "nvidia-smi not available\n")

    # 清空日志文件
    open(LOG_FILE, "w").close()

    update_config("starting", node_name, job_id)

    # 加载模型
    load_model()

    # 启动 HTTP 服务器
    PORT = 8000
    server = HTTPServer(("0.0.0.0", PORT), APIHandler)

    if node_name:
        base_url = f"http://{node_name}:{PORT}"
    else:
        base_url = f"http://localhost:{PORT}"

    update_config("running", node_name, job_id, base_url)

    print("=" * 60)
    print(f"✅ Gemma 服务器就绪！")
    print(f"API 地址: {base_url}")
    print(f"节点: {node_name}")
    print(f"作业ID: {job_id}")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务器...")
        server.shutdown()
        update_config("stopped", node_name, job_id)
        print("服务器已停止")


if __name__ == "__main__":
    main()
