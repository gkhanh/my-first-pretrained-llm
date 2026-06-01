# Finance Domain

## Goal

The model should be a **simple, factual finance assistant** that:
- Reads and summarises SEC filings (10-K, 10-Q, 8-K) in plain language
- Answers basic finance vocabulary and concept questions
- Performs simple numerical reasoning over reported figures
- Refuses to give investment advice

It is **not** a Bloomberg terminal, a financial forecasting model, or a trading signal generator.

## Data sources

### 1. SEC EDGAR Public Filings (primary)

| Type | Description |
|---|---|
| 10-K | Annual report — revenue, expenses, risk factors, MD&A |
| 10-Q | Quarterly report — YoY comparisons, interim financials |
| 8-K | Current report — material events (earnings releases, mergers, etc.) |

**Access**: Public API at `https://data.sec.gov/submissions/`. No authentication required. Fully permissive for research use.

**Filtering**: Focus on S&P 500 filings from 2000–present for high-quality financial English. Strip HTML tags, keep text sections (business description, risk factors, MD&A). Exclude financial tables (too sparse for pretraining).

**Volume**: ~500K filings → ~2–4 GB of cleaned text.

Download and prepare:
```bash
python scripts/data/prepare_finance_corpus.py \
    --source sec-edgar \
    --output-dir data/raw/finance/edgar/ \
    --start-year 2000 \
    --form-types 10-K 10-Q 8-K
```

### 2. Financial news (FNSPID or similar)

**FNSPID** (Financial News and Stock Price Integration Dataset) or a similar permissively-licensed news dataset.

Check current availability at HuggingFace:
```bash
# Search for permissive financial news datasets
python -c "from datasets import load_dataset; ds = load_dataset('Zihan1004/FNSPID')"
```

If FNSPID is not available or its license is unclear, fall back to:
- Financial articles from **CC-News** (filtered to finance topics via keyword tagging)
- **Reuters news corpus** subsets that are permissively released

### 3. StackExchange — Money / Quant / Personal Finance

Available as HuggingFace dataset dumps. High signal — real human questions with expert answers.

```bash
python scripts/data/prepare_finance_corpus.py \
    --source stackexchange \
    --subforums money quant personal-finance \
    --output-dir data/raw/finance/stackexchange/
```

## Licensing policy

**Rule**: Only use sources that are either public domain, CC-BY-compatible, or explicitly permissive for research.

| Source | License | Status |
|---|---|---|
| SEC EDGAR filings | Public domain (US government) | ✅ Use freely |
| FNSPID | Check dataset card | ⚠️ Verify before use |
| CC-News financial subset | CC | ✅ Use freely |
| StackExchange dumps | CC BY-SA 4.0 | ✅ Use freely (credit) |
| Bloomberg, Reuters paywalled | Non-permissive | ❌ Do not use |
| Financial Times, WSJ | Non-permissive | ❌ Do not use |

The goal is a model that is **legally redistributable** — only use sources that allow model weights trained on them to be shared.

## Evaluation

Two automated benchmarks:

### FinQA

Numerical reasoning over financial question-answer pairs derived from S&P 500 earnings reports.
- 8,281 examples (train: 6,251 / dev: 883 / test: 1,147)
- Metric: **execution accuracy** (does the model produce the correct numeric value via a chain-of-computation program?)
- HuggingFace: `ibm/finqa`

### ConvFinQA

Multi-turn conversational version of FinQA.
- 3,892 examples
- Tests the model's ability to track context and follow-up questions across a multi-turn dialogue
- HuggingFace: `ibm/convfinqa`

Run both:
```bash
python scripts/eval/finance_qa.py \
    --benchmark finqa convfinqa \
    --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
    --split test
```

## What the model should refuse

The model is trained to **refuse** investment advice prompts. Sample refusal training examples in `data/sft/refusals.jsonl`:

```json
{"messages": [
  {"role": "user", "content": "Should I buy $NVDA stock tomorrow?"},
  {"role": "assistant", "content": "I can provide factual information about NVIDIA's recent financial results, but I'm not able to give investment advice. For investment decisions, consult a licensed financial advisor."}
]}
```

Refusal is taught via a small set (~200 examples) included in the Stage-1 SFT mix. It is not enforced with RLHF — this is a best-effort safety measure, not a production safety system.

## Domain weight tuning

Default pretrain mix: **15% finance** (8% EDGAR + 5% news + 2% StackExchange).

If held-out finance perplexity is not improving after 150B tokens:
1. Raise finance weight to 25% in `configs/data/pretrain_mix.yaml`.
2. Resume from latest checkpoint (no need to restart — the data pipeline handles mid-run weight changes).
3. Reduce code weight from 65% to 55% to compensate.
