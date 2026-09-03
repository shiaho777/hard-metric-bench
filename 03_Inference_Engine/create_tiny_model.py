from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


def main() -> None:
    base = Path(__file__).resolve().parent / "models" / "qwen3-0.6b"
    base.mkdir(parents=True, exist_ok=True)

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "You": 2,
        "are": 3,
        "a": 4,
        "benchmark": 5,
        "prompt": 6,
        ".": 7,
        "Return": 8,
        "concise": 9,
        "text": 10,
        "only": 11,
    }

    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(base)

    config = GPT2Config(
        vocab_size=len(vocab),
        n_positions=1024,
        n_ctx=1024,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(base, safe_serialization=True)
    print(base)


if __name__ == "__main__":
    main()
