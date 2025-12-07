Fine-tuning / LoRA plan
=======================

Objectives
----------
- Specialize the assistant for funnel-drop troubleshooting, reducing manual ops escalations.
- Keep JSON response contracts stable while improving recall and nudge helpfulness.
- Preserve multilingual capability (English + Hinglish/Hindi) and compliance-safe nudges.

Dataset design
--------------
- Sources: existing troubleshooting chunks (VKYC, PAN upload, OTP, document readiness), error-code guides, and nudge prompt patterns.
- Composition targets (after expansion):
  - 45% funnel troubleshooting dialogues covering stage-specific flows (VKYC, PAN scan, OTP, document checklist) with retrieved chunk IDs.
  - 25% error-code focused turns (OCR_FAIL, PAN_GLARE, WINDOW_CLOSED, OTP_TIMEOUT/RETRY) with precise grounding.
  - 20% nudge-focused turns (CTA + explanatory) with compliance attributes and style constraints.
  - 10% multilingual variants (English/Hinglish/Hindi) spanning the above cases; prefer translation + light edits over direct MT to keep tone.
- Format: chat-style SFT with `system` guardrails, `user` containing `user_state` + free-text query, `assistant` emitting JSON contract: `predicted_drop_reason`, `explanation`, `steps_to_fix`, `confidence_score`, `citations`, `nudge_messages`, `conversational_message`.
- Quality controls: deduplicate near-clones, cluster by stage/error to avoid leakage across train/val/test, keep at least 10% validation for early stopping.
- Negative/edge cases: include low-confidence retrievals, missing error codes, off-hours VKYC, glare/angle PAN failures, OTP retry thresholds, and “no query supplied” fallbacks.

Model choice
------------
- Base: Llama-3.1-8B-Instruct (8k context). Fits single A100/48GB or dual 24GB cards with LoRA; retains multilingual support and fast inference for production.
- Rationale: strong instruction following, open weights for on-prem, good JSON adherence, and compatible with QLoRA for memory efficiency.

Hyperparameters (LoRA)
----------------------
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.
- Rank / alpha / dropout: r=64, alpha=128, dropout=0.1.
- Max seq len: 2048 tokens; pack multiple short turns to improve utilization.
- Optimizer: AdamW, lr=2e-4, weight decay=0.01, cosine decay, warmup=5%.
- Batch: effective ~256 sequences via gradient accumulation (e.g., micro-batch 8, accumulation 32), gradient clip 1.0, bf16.
- Epochs: 3 (monitor val loss + JSON correctness; early stop patience=1).
- Checkpointing: gradient checkpoint + FlashAttention to control memory; save best + final. Merge LoRA weights for deployment artifact.

Training & data pipeline
------------------------
- Apply a deterministic chat template with clear system role and fenced JSON schema; enforce quoting for citations.
- Balance sampling to avoid over-fitting VKYC/OTP at the expense of PAN/doc readiness.
- Generate light synthetic variants: paraphrases, reordered steps, and multilingual rewrites with human spot checks on 10% sample.
- Validate each record for JSON validity and presence of mandatory fields before training.

Evaluation plan
---------------
- Metrics: recall@5 (retrieval alignment), hallucination rate (citation-in-source), JSON correctness (schema presence), nudge helpfulness (LLM-judge score 1–5), avg latency, avg token cost.
- Slices: by stage (VKYC, OTP, PAN, docs), by error code, by language.
- Guardrail checks: profanity/PII leakage scan on generated nudges and conversational text.

Before/after expectations
-------------------------
- Baseline (current synthetic run): recall@5 ≈ 0.83, hallucination 0.0, JSON correctness 1.0, nudge helpfulness ≈ 4.27, avg latency ~0.98s, avg token cost ~0.0045.
- Post-fine-tune targets: recall@5 ≥ 0.90 on seen stages and ≥ 0.80 on new edge cases; maintain hallucination ≤ 0.02 and JSON correctness ≥ 0.98; nudge helpfulness ≥ 4.4 without latency regression.
- Success is defined as hitting targets across VKYC hours, PAN glare, OTP retry, and multilingual slices without increasing guardrail violations.

