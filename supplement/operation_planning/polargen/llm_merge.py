"""Merge the crystallography and operation-ranking LoRA adapters in order."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--crystal-adapter", type=Path, required=True)
    parser.add_argument("--rank-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="4GB")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model.resolve()),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        model,
        str(args.crystal_adapter.resolve()),
        is_trainable=False,
        local_files_only=True,
    ).merge_and_unload()
    model = PeftModel.from_pretrained(
        model,
        str(args.rank_adapter.resolve()),
        is_trainable=False,
        local_files_only=True,
    ).merge_and_unload()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(output),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.rank_adapter.resolve()),
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(str(output))
    print(f"PASS merged_model={output}")


if __name__ == "__main__":
    main()
