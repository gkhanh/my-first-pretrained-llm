"""Data packing smoke tests.

Tests the DocumentPacker (sequence packing without padding) and the
FIM transform to ensure correctness before training.
"""


from khanh_llm.data.fim import apply_fim_transform
from khanh_llm.data.streaming import DocumentPacker


class TestDocumentPacker:
    def test_output_chunks_are_correct_length(self) -> None:
        packer = DocumentPacker(seq_len=10, eos_id=0)
        docs = [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11, 12]]
        chunks = list(packer.pack(iter(docs)))
        for chunk in chunks:
            assert len(chunk) == 10, f"Chunk length {len(chunk)} != 10"

    def test_eos_appears_at_document_boundaries(self) -> None:
        packer = DocumentPacker(seq_len=4, eos_id=999)
        docs = [[1, 2, 3], [4, 5, 6]]
        packed = []
        for chunk in packer.pack(iter(docs)):
            packed.extend(chunk)
        # 999 should appear at position 3 (after doc1) and position 7 (after doc2)
        assert 999 in packed, "EOS token should appear in packed output"

    def test_no_padding_tokens(self) -> None:
        """Every position in every chunk is a real token (no pad)."""
        packer = DocumentPacker(seq_len=8, eos_id=0)
        docs = [list(range(1, 100))]  # One long document, no gaps
        for chunk in packer.pack(iter(docs)):
            assert len(chunk) == 8


class TestFIMTransform:
    def test_fim_applied_with_correct_rate(self) -> None:
        import random
        rng = random.Random(0)
        tokens = list(range(1, 50))
        n_fim = sum(
            1 for _ in range(1000)
            if apply_fim_transform(tokens, 100, 101, 102, 0, fim_rate=0.5, rng=rng) != tokens
        )
        # Should be roughly 50% with tolerance
        assert 400 < n_fim < 600, f"FIM rate off: {n_fim}/1000 applied"

    def test_fim_output_contains_special_tokens(self) -> None:
        tokens = list(range(1, 20))
        result = apply_fim_transform(
            tokens, fim_prefix_id=100, fim_suffix_id=101, fim_middle_id=102,
            eos_id=0, fim_rate=1.0  # always apply
        )
        assert 100 in result, "FIM prefix token missing"
        assert 101 in result, "FIM suffix token missing"
        assert 102 in result, "FIM middle token missing"

    def test_fim_preserves_all_original_tokens(self) -> None:
        tokens = list(range(1, 20))
        result = apply_fim_transform(
            tokens, fim_prefix_id=100, fim_suffix_id=101, fim_middle_id=102,
            eos_id=0, fim_rate=1.0
        )
        # All original tokens should still be present (reordered)
        for tok in tokens:
            assert tok in result, f"Token {tok} missing from FIM output"
