"""KV-cache-backed text generator with streaming support.

Replaces the O(n²) generation loop in the legacy generate_text.py with a proper
per-step KV-cache update, reducing inference cost from O(n²) to O(n).

Usage:
    from khanh_llm.inference.generator import Generator

    gen = Generator.from_checkpoint("runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt")
    for token_text in gen.stream("def fibonacci(n):"):
        print(token_text, end="", flush=True)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from khanh_llm.config import ModelConfig
from khanh_llm.inference.samplers import repetition_penalty_logits, top_k_filter, top_p_filter
from khanh_llm.model.transformer import KhanhLLM
from khanh_llm.training.checkpoint import _strip_compile_prefix


class Generator:
    """KV-cache-backed autoregressive text generator.

    Args:
        model: KhanhLLM instance in eval mode.
        tokenizer: HuggingFace tokenizer.
        device: Torch device.
        compile_model: Whether to torch.compile the model for inference.
    """

    def __init__(self, model: KhanhLLM, tokenizer, device: torch.device | str, compile_model: bool = True) -> None:
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = torch.device(device)
        self.model.eval()
        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="default")
            except Exception as e:
                print(f"[Generator] torch.compile skipped: {e}")

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        tokenizer_path: str | Path,
        model_cfg: ModelConfig | None = None,
        device: str | torch.device = "cuda",
        use_ema: bool = True,
        compile_model: bool = True,
    ) -> Generator:
        """Load a generator from a training checkpoint.

        Args:
            ckpt_path: Path to the .pt checkpoint or the EMA weights file.
            tokenizer_path: Path to the tokenizer directory.
            model_cfg: ModelConfig (loaded from checkpoint if None).
            device: Target device.
            use_ema: If True, tries to load EMA weights from the adjacent ema.pt file.
            compile_model: Whether to compile for inference.
        """
        from transformers import AutoTokenizer

        ckpt_path = Path(ckpt_path)
        device    = torch.device(device)

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Build model config
        if model_cfg is None and "config" in ckpt:
            cfg_dict   = ckpt["config"].get("model", {})
            model_cfg  = ModelConfig(**{k: v for k, v in cfg_dict.items() if hasattr(ModelConfig, k)})
        if model_cfg is None:
            model_cfg = ModelConfig()

        model = KhanhLLM(model_cfg).to(device)

        # Try EMA weights first (better inference quality)
        ema_path = ckpt_path.parent / "trained_khanh_gpt_ema.pt"
        if use_ema and ema_path.exists():
            ema_ckpt = torch.load(ema_path, map_location=device, weights_only=False)
            sd = _strip_compile_prefix({k: v.to(device) for k, v in ema_ckpt["ema_state_dict"].items()})
            print(f"[Generator] Loaded EMA weights from {ema_path}")
        else:
            sd = _strip_compile_prefix(ckpt["model_state_dict"])
            print(f"[Generator] Loaded weights from {ckpt_path}")

        # Tied embeddings: output.weight == token_embedding.weight; fill in if absent
        if "output.weight" not in sd and "token_embedding.weight" in sd:
            sd["output.weight"] = sd["token_embedding.weight"]

        model.load_state_dict(sd, strict=True)

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        return cls(model, tokenizer, device=device, compile_model=compile_model)

    @classmethod
    def from_legacy_checkpoint(
        cls,
        ckpt_path: str | Path,
        tokenizer_path: str | Path,
        device: str | torch.device = "cuda",
    ) -> Generator:
        """Load the legacy 692M MoE checkpoint for backward compatibility.

        This uses the OLD KhanhLLM class from models/khanh_llm.py (legacy).
        The generator wraps it in a shim so the same .stream() / .generate() API works.
        """
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).parents[4]))  # repo root
        from models.khanh_llm import KhanhLLM as LegacyKhanhLLM  # type: ignore
        from transformers import AutoTokenizer

        device = torch.device(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        legacy_model = LegacyKhanhLLM().to(device)
        sd = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in ckpt["model_state_dict"].items()
        }
        legacy_model.load_state_dict(sd, strict=True)
        legacy_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # Wrap in a thin shim so the Generator API works
        return _LegacyGeneratorShim(legacy_model, tokenizer, device)

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.6,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: int | None = None,
    ) -> Iterator[str]:
        """Generate tokens one at a time, yielding decoded text for each.

        This uses the KV-cache so each step is O(1) rather than O(n).

        Args:
            prompt: Text prompt to condition on.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (lower = more focused).
            top_k: Top-k filtering (0 = disabled).
            top_p: Nucleus sampling threshold.
            repetition_penalty: Penalty for repeating recent tokens (1.0 = off).
            eos_token_id: Stop when this token is generated (defaults to tokenizer EOS).

        Yields:
            Decoded text string for each generated token.
        """
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Prefill: run the full prompt through the model once to build the KV cache
        _, _, past_kvs = self.model(input_ids)

        all_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Only pass the last token on subsequent steps — KV cache handles the rest
            last_token = all_ids[:, -1:]
            logits, _, past_kvs = self.model(last_token, past_kvs=past_kvs)

            next_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Sampling filters
            next_logits = repetition_penalty_logits(next_logits, all_ids, repetition_penalty)
            next_logits = top_k_filter(next_logits, top_k)
            next_logits = top_p_filter(next_logits, top_p)

            probs      = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            token_id = next_token.item()
            all_ids  = torch.cat([all_ids, next_token], dim=1)

            # Yield the decoded text for this token
            yield self.tokenizer.decode([token_id], skip_special_tokens=True)

            if eos_token_id is not None and token_id == eos_token_id:
                break

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        **kwargs,
    ) -> tuple[str, str]:
        """Blocking generation. Returns (full_text, generated_text).

        Convenience wrapper over stream() for non-streaming usage.
        """
        tokens = []
        for tok_text in self.stream(prompt, max_new_tokens=max_new_tokens, **kwargs):
            tokens.append(tok_text)
        generated = "".join(tokens)
        return prompt + generated, generated


class _LegacyGeneratorShim:
    """Thin shim wrapping the legacy 692M model in the Generator interface.

    NOTE: No KV cache — uses the old O(n²) loop. This is intentional:
    the legacy model does not support KV cache and is only used for
    loading archived checkpoints via --legacy flag.
    """

    def __init__(self, legacy_model, tokenizer, device: torch.device) -> None:
        self.model     = legacy_model
        self.tokenizer = tokenizer
        self.device    = device

    @torch.inference_mode()
    def stream(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.6,
               top_k: int = 40, top_p: float = 0.9, repetition_penalty: float = 1.1,
               eos_token_id: int | None = None) -> Iterator[str]:
        from khanh_llm.inference.samplers import (
            repetition_penalty_logits,
            top_k_filter,
            top_p_filter,
        )

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        all_ids   = input_ids.clone()

        for _ in range(max_new_tokens):
            logits, _ = self.model(all_ids)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            next_logits = repetition_penalty_logits(next_logits, all_ids, repetition_penalty)
            next_logits = top_k_filter(next_logits, top_k)
            next_logits = top_p_filter(next_logits, top_p)
            probs       = torch.softmax(next_logits, dim=-1)
            next_token  = torch.multinomial(probs, num_samples=1)
            token_id    = next_token.item()
            all_ids     = torch.cat([all_ids, next_token], dim=1)
            yield self.tokenizer.decode([token_id], skip_special_tokens=True)
            if eos_token_id is not None and token_id == eos_token_id:
                break

    def generate(self, prompt: str, max_new_tokens: int = 200, **kwargs) -> tuple[str, str]:
        tokens = list(self.stream(prompt, max_new_tokens=max_new_tokens, **kwargs))
        generated = "".join(tokens)
        return prompt + generated, generated
