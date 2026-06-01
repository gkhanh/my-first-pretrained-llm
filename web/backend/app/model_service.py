"""ModelService — wraps khanh_llm.inference.generator for the FastAPI backend.

Method signatures are defined; implementation is deferred to Phase 4 proper.
See docs/06-web-app-design.md for the design contract.
"""

from __future__ import annotations

from collections.abc import AsyncIterator


class ModelService:
    """Singleton wrapper that holds the loaded Generator instance.

    Call load() once at startup (e.g. in a FastAPI lifespan event).
    generate() and stream_generate() are called per request.
    """

    _instance: ModelService | None = None

    def __new__(cls) -> ModelService:
        # Singleton — all routes share one ModelService
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._generator = None
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self._generator is not None

    def load(self, ckpt_path: str, tokenizer_path: str, device: str = "cuda") -> None:
        """Load the KhanhLLM generator from a checkpoint.

        Args:
            ckpt_path: Path to the .pt checkpoint file.
            tokenizer_path: Path to the StarCoder2 tokenizer directory.
            device: Torch device string.
        """
        raise NotImplementedError("see docs/06-web-app-design.md")

    def generate(self, prompt: str, **kwargs) -> tuple[str, int]:
        """Blocking generation. Returns (generated_text, tokens_generated).

        Args:
            prompt: Input prompt.
            **kwargs: Forwarded to Generator.generate().
        """
        raise NotImplementedError("see docs/06-web-app-design.md")

    async def stream_generate(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Async streaming generation. Yields one token string at a time.

        Args:
            prompt: Input prompt.
            **kwargs: Forwarded to Generator.stream().
        """
        raise NotImplementedError("see docs/06-web-app-design.md")
        yield  # make this a generator function so the signature is correct
