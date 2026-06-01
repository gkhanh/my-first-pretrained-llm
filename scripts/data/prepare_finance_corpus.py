"""
Download and clean finance-domain text for pretraining.

Sources (all permissively licensed / public domain):
  - SEC EDGAR 10-K, 10-Q, 8-K filings (public domain, sec.gov/Archives)
  - FNSPID financial news dataset (permissive license)
  - StackExchange Money, Quant, Personal Finance (CC BY-SA 4.0)

Output: cleaned .jsonl files under data/raw/finance/, one line per document:
    {"text": "...", "source": "edgar|fnspid|stackexchange", "id": "..."}

These are then picked up by prepare_pretrain_corpus.py via configs/data/pretrain_mix.yaml.

Usage:
    python -m scripts.data.prepare_finance_corpus --out_dir data/raw/finance
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/raw/finance")
    p.add_argument("--sources", nargs="+",
                   default=["edgar", "fnspid", "stackexchange"],
                   choices=["edgar", "fnspid", "stackexchange"])
    p.add_argument("--edgar_years", nargs="+", type=int,
                   default=list(range(2010, 2024)),
                   help="Which filing years to download from EDGAR")
    p.add_argument("--max_docs", type=int, default=None,
                   help="Cap per source (useful for smoke tests)")
    return p.parse_args()


def download_edgar(out_dir: Path, years: list[int], max_docs: int | None):
    """Download SEC EDGAR full-text index and fetch filing text."""
    raise NotImplementedError(
        "EDGAR download not yet implemented. "
        "See docs/09-finance-domain.md for the filing index URL format: "
        "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/company.idx"
    )


def download_fnspid(out_dir: Path, max_docs: int | None):
    """Download FNSPID financial news from HuggingFace datasets."""
    raise NotImplementedError(
        "FNSPID download not yet implemented. "
        "Run: from datasets import load_dataset; ds = load_dataset('Zihan1004/FNSPID')"
    )


def download_stackexchange_finance(out_dir: Path, max_docs: int | None):
    """Download Money/Quant/PersonalFinance StackExchange dumps."""
    raise NotImplementedError(
        "StackExchange finance download not yet implemented. "
        "See docs/09-finance-domain.md for the archive.org dump URLs."
    )


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dispatch = {
        "edgar": lambda: download_edgar(out_dir / "edgar", args.edgar_years, args.max_docs),
        "fnspid": lambda: download_fnspid(out_dir / "fnspid", args.max_docs),
        "stackexchange": lambda: download_stackexchange_finance(out_dir / "stackexchange", args.max_docs),
    }

    for source in args.sources:
        print(f"Downloading {source}...")
        dispatch[source]()
        print(f"  {source} done.")

    print(f"Finance corpus written to {out_dir}")
    print("Next: run prepare_pretrain_corpus.py to tokenize and pack.")


if __name__ == "__main__":
    main()
