"""Memory-mapped pre-tokenized shard utilities.

Shards are flat binary files of uint16 token IDs.
Each shard has an accompanying .idx file of uint64 byte offsets (one per document).

File format:
    shard_NNNN.bin   — flat uint16 array of token IDs
    shard_NNNN.idx   — flat uint64 array of document start offsets (in tokens)

Usage:
    from khanh_llm.data.shards import ShardedDataset

    ds = ShardedDataset("data/shards/pretrain", seq_len=2048)
    for tokens, labels in ds:
        ...
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

DTYPE = np.uint16   # 49K vocab fits in uint16
DTYPE_BYTES = 2


class ShardWriter:
    """Writes pre-tokenized token IDs to a binary shard file.

    Args:
        path: Path to the output .bin file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")
        self._idx_fh = open(self.path.with_suffix(".idx"), "wb")
        self._offset = 0

    def write_document(self, token_ids: list[int]) -> None:
        """Write a single document's token IDs and record its start offset."""
        # Record document start offset (in tokens, not bytes)
        self._idx_fh.write(struct.pack("<Q", self._offset))

        arr = np.array(token_ids, dtype=DTYPE)
        self._fh.write(arr.tobytes())
        self._offset += len(token_ids)

    def close(self) -> None:
        self._fh.close()
        self._idx_fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class ShardedDataset(IterableDataset):
    """IterableDataset over pre-tokenized .bin shard files.

    Reads shards in order, packs tokens into seq_len chunks, and
    yields (input_ids, labels) pairs where labels = input_ids shifted left by 1.

    Args:
        shard_dir: Directory containing shard_NNNN.bin files.
        seq_len: Sequence length (chunk size).
        start_shard: Index of the first shard to read (for resuming).
        start_offset: Token offset within the first shard (for resuming).
    """

    def __init__(
        self,
        shard_dir: str | Path,
        seq_len: int = 2048,
        start_shard: int = 0,
        start_offset: int = 0,
    ) -> None:
        super().__init__()
        self.shard_dir    = Path(shard_dir)
        self.seq_len      = seq_len
        self.start_shard  = start_shard
        self.start_offset = start_offset

        self.shard_paths = sorted(self.shard_dir.glob("shard_*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"No shard files found in {shard_dir}. "
                "Run: python scripts/data/prepare_pretrain_corpus.py"
            )

    def __iter__(self):
        buf = np.empty(0, dtype=DTYPE)

        # Each DataLoader worker gets a disjoint subset of shards so no data is repeated.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shard_paths = [p for i, p in enumerate(self.shard_paths)
                           if i % worker_info.num_workers == worker_info.id]
        else:
            shard_paths = self.shard_paths

        for shard_path in shard_paths:
            shard_idx = self.shard_paths.index(shard_path)
            if shard_idx < self.start_shard:
                continue

            data = np.memmap(shard_path, dtype=DTYPE, mode="r")
            offset = self.start_offset if shard_idx == self.start_shard else 0
            data = data[offset:]

            buf = np.concatenate([buf, data])

            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf   = buf[self.seq_len :]

                input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
                labels    = torch.from_numpy(chunk[1:].astype(np.int64))
                yield input_ids, labels
