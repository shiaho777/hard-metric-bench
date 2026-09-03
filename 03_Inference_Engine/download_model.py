from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    target_dir = base_dir / "models" / "qwen3-0.6b"
    repo_id = "Qwen/Qwen3-0.6B"
    print(f"准备下载 {repo_id} -> {target_dir}")
    print("说明: 当前官方公开的最小 Qwen3 dense 模型为 0.6B，本脚本以它作为统一测试模型。")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=["*.json", "*.model", "*.tiktoken", "*.txt", "*.safetensors", "*.py"],
        )
    except Exception as exc:
        print(f"下载失败: {exc}")
        print("如需镜像，可设置 HF_ENDPOINT=https://hf-mirror.com")
        return 1
    metadata = {
        "repo_id": repo_id,
        "local_dir": str(target_dir),
    }
    (target_dir / "download_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("下载完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
