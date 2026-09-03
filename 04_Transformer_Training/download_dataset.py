from __future__ import annotations

import json
from pathlib import Path

TARGET_BYTES = 10 * 1024 * 1024
PRIMARY_DATASET = "ddrg/named_math_formulas"
FALLBACK_DATASET = "OleehyO/latex-formulas"


def write_corpus(stream, text_key: str, target_file: Path) -> int:
    written = 0
    seen = set()
    with target_file.open("w", encoding="utf-8") as handle:
        for row in stream:
            text = str(row.get(text_key, "")).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            line = text.replace("\n", " ").strip()
            if len(line) < 5:
                continue
            payload = line + "\n"
            encoded = payload.encode("utf-8")
            if written + len(encoded) > TARGET_BYTES:
                break
            handle.write(payload)
            written += len(encoded)
    return written


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    target_file = dataset_dir / "math_formulas.txt"
    metadata_file = dataset_dir / "source_metadata.json"
    try:
        from datasets import load_dataset
    except Exception:
        print("请先安装 datasets: pip install datasets")
        return 1

    sources = [
        (PRIMARY_DATASET, "formula", {"split": "train", "streaming": True}),
        (FALLBACK_DATASET, "latex_formula", {"split": "train", "streaming": True}),
    ]
    errors = []
    for dataset_name, key, kwargs in sources:
        print(f"尝试下载数据集: {dataset_name}")
        try:
            stream = load_dataset(dataset_name, **kwargs)
            written = write_corpus(stream, key, target_file)
            if written >= TARGET_BYTES * 0.95:
                metadata = {
                    "dataset": dataset_name,
                    "field": key,
                    "bytes_written": written,
                    "target_bytes": TARGET_BYTES,
                    "output": str(target_file),
                }
                metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(f"数据集准备完成: {written / (1024 * 1024):.2f} MB -> {target_file}")
                return 0
            errors.append(f"{dataset_name} 仅写入 {written} bytes")
        except Exception as exc:
            errors.append(f"{dataset_name}: {exc}")
    print("数据集下载失败。")
    for item in errors:
        print(f"- {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
