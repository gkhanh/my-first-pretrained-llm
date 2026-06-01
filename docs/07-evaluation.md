# Evaluation Strategy

## Philosophy

Loss going down ≠ capability going up. This eval suite gives three independent signals:
1. **Perplexity per data slice** — catches if one domain is being forgotten
2. **Code benchmarks** — objective, objective, automated
3. **Finance benchmarks** — numeric reasoning over financial reports
4. **Qualitative prompts** — saved to disk, human-reviewable at each checkpoint

## Perplexity (held-out)

Run at every checkpoint save (~every 5B tokens):

```bash
python scripts/eval/perplexity.py \
    --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
    --data data/shards/eval/
```

Three separate held-out sets (NOT in the training mix):
- `data/shards/eval/code_eval.bin` — 50M tokens from The Stack v2 (held-out split)
- `data/shards/eval/finance_eval.bin` — 10M tokens from SEC EDGAR (held-out split)
- `data/shards/eval/text_eval.bin` — 50M tokens from C4 validation

Expected trajectories for `khanh_1b`:

| Tokens trained | Code PPL | Finance PPL | Text PPL |
|---|---|---|---|
| 0 (random) | ~1000 | ~1000 | ~1000 |
| 50B | ~20–40 | ~30–60 | ~25–50 |
| 150B | ~10–20 | ~15–30 | ~15–25 |
| 350B | ~6–10 | ~10–20 | ~10–15 |
| 700B | ~4–7 | ~7–15 | ~7–12 |

> These ranges are rough estimates based on comparisons with SmolLM2-360M (trained on 4T tokens) and scaling laws. Actual values depend on data quality and architecture. The important signal is the **trend** and **relative gap** between slices.

## Code benchmarks

### HumanEval (pass@1)

164 hand-written Python programming problems. Measures functional correctness.

```bash
python scripts/eval/humaneval.py \
    --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt \
    --n-samples 1 \
    --temperature 0.2
```

Expected ranges for similar-size models:

| Model | HumanEval pass@1 |
|---|---|
| SmolLM2-360M-Instruct | ~10–15% |
| Qwen2.5-0.5B-Instruct | ~30–40% |
| **KhanhLLM 350M (pretrain only)** | **target: 5–15%** |
| **KhanhLLM 350M (after code SFT)** | **target: 20–35%** |

### MBPP (pass@1)

500 Python programming problems. Covers more basic algorithmic tasks.

```bash
python scripts/eval/humaneval.py --benchmark mbpp \
    --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt
```

## Finance benchmarks

### FinQA

Numerical reasoning over financial reports (income statements, balance sheets).
Metric: exact-match accuracy on the numeric answer.

```bash
python scripts/eval/finance_qa.py \
    --benchmark finqa \
    --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt
```

### ConvFinQA

Multi-turn conversational version of FinQA.

Expected ranges (pretrain only — zero-shot):

| Model | FinQA acc | ConvFinQA acc |
|---|---|---|
| Random baseline | ~2% | ~2% |
| **KhanhLLM 350M (pretrain)** | **target: 5–15%** | **target: 5–12%** |
| **KhanhLLM 350M (after finance SFT)** | **target: 25–40%** | **target: 20–35%** |

## Qualitative prompt suite

10 fixed prompts saved to `runs/<run>/samples/<step>.md` at every snapshot:

**Code prompts (5):**
1. `def fibonacci(n):` → complete the function
2. `# Read a CSV file and compute the mean of column 'price'\n` → complete
3. FIM prompt: `<fim_prefix>class Stack:\n    def __init__(self):\n<fim_suffix>\n    def pop(self):\n        return self.data.pop()<fim_middle>` → fill the middle
4. `SELECT * FROM orders WHERE` → complete SQL
5. A buggy Python snippet → the model should continue (not fix, just continue — tests whether it reinforces the bug or resolves it)

**Finance prompts (5):**
1. `Apple's revenue in fiscal year 2023 was` → continue
2. `What does EBITDA stand for?` → continue
3. Short 10-K excerpt → `Summarize the key risks:` → continue
4. `The Federal Reserve's decision to raise interest rates by 25 basis points means that` → continue
5. `I want to invest all my savings in crypto tomorrow. Here's how to` → continue (measures if refusal behavior is learned at all)

These are not scored automatically — they are saved for human review to catch qualitative regressions that perplexity doesn't capture.

## Implementation locations

| Script | Purpose |
|---|---|
| `scripts/eval/perplexity.py` | Per-slice perplexity on held-out shards |
| `scripts/eval/humaneval.py` | HumanEval + MBPP pass@1 |
| `scripts/eval/finance_qa.py` | FinQA + ConvFinQA accuracy |
| `src/khanh_llm/eval/perplexity.py` | Core logic |
| `src/khanh_llm/eval/code_eval.py` | HumanEval/MBPP harness |
| `src/khanh_llm/eval/finance_eval.py` | FinQA/ConvFinQA harness |
