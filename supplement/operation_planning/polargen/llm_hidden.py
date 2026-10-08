"""Extract the exact decoder-layer states used by the language-guided ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_record(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("Language input is empty")
    if "\n" not in text:
        return json.loads(text)
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("This executable accepts exactly one input record")
    return rows[0]


def render(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_input_tokens: int,
) -> tuple[list[int], bool]:
    if [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("input_messages must be system then user")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]
    if len(input_ids) <= max_input_tokens:
        return input_ids, False
    return input_ids[-max_input_tokens:], True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Operation-ranking LoRA. Omit when --base-model is fully merged.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[15, 47])
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    args = parser.parse_args()

    record = read_record(args.input.resolve())
    messages = record.get("input_messages") or record.get(
        "llm_input_messages"
    )
    if messages is None:
        raise ValueError("Record has no input_messages")
    tokenizer_source = (
        args.adapter.resolve() if args.adapter is not None
        else args.base_model.resolve()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_source),
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model.resolve()),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(
            base,
            str(args.adapter.resolve()),
            is_trainable=False,
            local_files_only=True,
        )
    else:
        model = base
    model.eval()
    backbone = (
        model.get_base_model().model
        if isinstance(model, PeftModel)
        else model.model
    )
    decoder_layers = getattr(backbone, "layers", None)
    if decoder_layers is None:
        raise RuntimeError("Qwen decoder layers were not found")
    for index in args.layers:
        if index < 0 or index >= len(decoder_layers):
            raise ValueError(f"Invalid decoder layer {index}")
    capture: dict[int, np.ndarray] = {}

    def make_hook(index: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            capture[index] = (
                hidden[:, -1, :].detach().to(torch.float16).cpu().numpy()
            )

        return hook

    hooks = [
        decoder_layers[index].register_forward_hook(make_hook(index))
        for index in args.layers
    ]
    input_ids, trimmed = render(
        tokenizer, messages, args.max_input_tokens
    )
    first_device = model.get_input_embeddings().weight.device
    input_tensor = torch.tensor(
        [input_ids], dtype=torch.long, device=first_device
    )
    attention_mask = torch.ones_like(input_tensor)
    try:
        with torch.inference_mode():
            model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        for hook in hooks:
            hook.remove()
    missing = [index for index in args.layers if index not in capture]
    if missing:
        raise RuntimeError(f"Decoder hooks did not fire: {missing}")
    payload: dict[str, Any] = {
        "id": np.asarray(str(record.get("id", "inference")), dtype=np.str_),
        "token_count": np.asarray(len(input_ids), dtype=np.int64),
        "trimmed": np.asarray(trimmed, dtype=np.bool_),
        "model_mode": np.asarray(
            "base_plus_adapter" if args.adapter is not None else "merged",
            dtype=np.str_,
        ),
    }
    for index in args.layers:
        payload[f"layer_{index}"] = capture[index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **payload)
    print(
        f"PASS id={record.get('id')} layers={args.layers} "
        f"tokens={len(input_ids)} output={args.output}"
    )


if __name__ == "__main__":
    main()
