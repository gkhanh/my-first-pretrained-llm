# 2-Year Roadmap (~6,000 GPU-Hours)

## Timeline overview

| Period | GPU hours | Milestone |
|---|---|---|
| Months 1–2 | ~290 h | Repo restructure, architecture rebuild, tokenizer + data pipeline, debug runs, fine-tuning pipeline |
| Months 3–22 | ~5,400 h | Full pretrain of `khanh_1b` on code+finance+text mix (~390B tokens) |
| Months 19–22 | ~750 h | Post-training: SFT (chat → code+finance staged), then DPO |
| Months 23–24 | ~290 h | Eval, polish, periodic LoRA top-ups for new domains |

## Months 1–2: Foundation (current phase)

**Goal**: Everything needed to start a clean pretrain run is in place.

- [x] Repo restructure (`src/khanh_llm/`, `configs/`, `scripts/`, `docs/`, `web/`, `tests/`)
- [ ] Architecture: RoPE, RMSNorm, SwiGLU, GQA+SDPA, pre-norm, tied embeddings, fixed MoE balance loss
- [ ] Training loop: bf16 autocast, sequence packing, pre-tokenized shards, EMA, eval hooks
- [ ] KV-cache inference + streaming generator (`chat_cli.py`)
- [ ] Tokenizer: Download StarCoder2 tokenizer to `data/tokenizers/starcoder2/`
- [ ] Data pipeline: `prepare_pretrain_corpus.py` — Stack v2 (Python/JS/TS/Go/Rust) + Wikipedia + C4 → sharded `.bin`
- [ ] Finance corpus: `prepare_finance_corpus.py` — SEC EDGAR + FNSPID + StackExchange Money
- [ ] Validate end-to-end: smoke-train `khanh_150m` for 1 hour, check throughput and loss curve
- [ ] Fine-tuning pipeline: LoRA + QLoRA on both KhanhLLM and external HF models
- [ ] QLoRA smoke test: fine-tune Qwen2.5-1.5B on a tiny code+finance SFT mix

> After validation, kick off the `khanh_1b` pretrain. This is the boundary between Months 2 and 3.

## Months 3–22: Pretrain `khanh_1b`

**Target**: ~390B tokens, loss < 3.0, meaningful code+finance capability.

**Token milestones** (approximate — depends on actual throughput):

| Tokens processed | Expected loss | Expected capability |
|---|---|---|
| 5B | ~7–8 | Random-ish, beginning of coherence |
| 50B | ~4–5 | Syntactically plausible text |
| 150B | ~3–3.5 | Basic code structure, some finance vocabulary |
| 300B | ~2.8–3.0 | SmolLM2-360M territory |
| 500B | ~2.5–2.8 | Qwen2.5-0.5B territory |
| 700B | ~2.3–2.5 | Target: "useful intern" for code + basic finance Q&A |

**Checkpoint schedule**: snapshot every ~5B tokens (~35h on 5080) → ~140 total, rolling 10 kept + permanent milestones every 10th.

**Intervention triggers**:
- Loss plateaus for > 50B tokens → check data pipeline, LR schedule, or reduce micro-batch to increase effective steps.
- Finance perplexity not improving → raise finance data weight from 15% to 25% in `configs/data/pretrain_mix.yaml` and resume from latest checkpoint.
- Expert collapse (if MoE ever enabled) → raise `aux_loss_weight`.

## Months 19–22: Post-Training

**Two-stage SFT** (avoids catastrophic forgetting):

**Stage 1 — General chat** (~1 epoch on UltraChat-200k subset + OpenHermes-2.5):
```bash
python scripts/finetune/sft.py \
    --config configs/train/sft_chat.yaml \
    --ckpt runs/khanh_1b/checkpoints/trained_khanh_gpt_latest.pt
```

**Stage 2 — Code + Finance** (~2 epochs on code+finance SFT mix, with 10% stage-1 replay):
```bash
python scripts/finetune/sft.py \
    --config configs/train/sft_code.yaml \
    --ckpt runs/khanh_1b_sft_chat/checkpoints/trained_khanh_gpt_latest.pt
```

**DPO** (optional, after SFT):
```bash
python scripts/finetune/dpo.py \
    --config configs/train/dpo.yaml \
    --ckpt runs/khanh_1b_sft_code/checkpoints/trained_khanh_gpt_latest.pt
```

## Months 23–24: Polish + Ongoing LoRA Top-Ups

- Run full eval suite: HumanEval, MBPP, FinQA, ConvFinQA, qualitative prompts.
- Add new programming languages via LoRA (each ~10 GPU-hours).
- Extend finance coverage (new datasets, more EDGAR filings) via LoRA.
- Optional: start a `khanh_700m` run on the new architecture if budget allows.

## Contingency: 3,000 GPU-Hours

Switch default config to `khanh_700m.yaml`. At ~40K tok/s and 58h/week uptime:
- 3,000h → ~600B tokens for a 250M model.
- A 250M model on 600B tokens of code+finance+text is competitive with same-size public models.
- Cut SFT to a single stage (mix chat+code+finance in one pass).
- Drop DPO.
- Keep the fine-tuning pipeline — it's cheap and high-leverage.

## Naming convention for runs

```
runs/
├── khanh_150m_debug/          ← validation run (Months 1–2)
├── khanh_1b/                ← the main 700B-token pretrain
├── khanh_1b_sft_chat/       ← after Stage 1 SFT
├── khanh_1b_sft_code/       ← after Stage 2 SFT
├── khanh_1b_dpo/            ← after DPO (optional)
└── _archive_legacy_692m/      ← the old 692M MoE checkpoint (preserved, not used)
```
