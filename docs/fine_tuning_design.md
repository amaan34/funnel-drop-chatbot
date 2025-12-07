# Fine-tuning / LoRA Design (No training executed)

## Objectives
- Improve grounding on funnel troubleshooting, error codes, and device/time nuances.
- Generate higher-quality multilingual nudges (English/Hindi) with compliance-safe tone.
- Reduce hallucinations via domain-constrained responses and JSON-correct outputs.

## Dataset Design
- Sources: curated Q&A from `data/Assignment RAG_context_file.pdf`, funnel steps, FAQs, troubleshooting snippets, and synthetic user states.
- Task types (balanced mix):
  - Drop-off reasoning: user_state → JSON {primary_reason, secondary_reasons, confidence, reasoning_chain}.
  - Nudge generation: (reason, stage, user_state) → 3 styles + compliance-safe variant, bilingual.
  - Error-code explanations: error_code + stage → concise cause + action steps.
  - Conversation turns: user query + retrieved context → grounded answer with citations placeholder tokens.
  - JSON adherence: outputs must be valid JSON only for reasoning tasks.
- Multilingual: 30–40% Hindi targets; rest English.
- Size: 5–10k examples (mix of synthetic + doc-derived). Keep validation split 10–15%.
- Formatting: chat-style messages (system/user/assistant) in JSONL; include citations fields when relevant.

## Model & Method
- Base: lightweight instruction-tuned model (e.g., Llama-3-8B-Instruct or Mistral-7B-Instruct).
- Adapter: LoRA/QLoRA to reduce memory and cost.
- Context window: prefer ≥8k tokens; truncation/packing strategy applied to inputs.

## Hyperparameters (starting point)
- Epochs: 3–5; Batch size: 64 tokens-per-device equivalent with gradient accumulation as needed.
- LR: 2e-4 (LoRA adapters); Scheduler: cosine with warmup 5%.
- LoRA: r=16, alpha=32, dropout=0.05; target modules: attention q_proj, k_proj, v_proj, o_proj.
- Max input tokens: 1024; Max output: 256 for reasoning, 180 for nudges.
- Weight decay: 0.01; Grad clip: 1.0; Label smoothing: 0.0.

## Evaluation Metrics
- RAG-grounding: recall@5 on held-out retrieval pairs.
- Hallucination: citation consistency rate.
- JSON correctness: % valid JSON for reasoning tasks.
- Nudge quality: LLM-as-judge (helpfulness/clarity/empathy/compliance).
- Multilingual quality: bilingual BLEU/COMET-lite on Hindi subset; manual spot checks.
- Latency/Cost: tokens-per-response and wall-clock latency on a small eval set.

## Before/After Expectations
- Reduce hallucination rate vs. baseline by ≥20%.
- Improve JSON correctness to ≥0.95 on reasoning tasks.
- Nudge LLM-judge score +0.3 to +0.5 absolute.
- Stable bilingual adequacy per spot checks; no regression in compliance flags.

## Training/Deployment Notes
- Use de-identified, synthetic augmentation; avoid PII.
- Keep adapters separate from base weights; version adapters with data/commit hash.
- Validate on a held-out, doc-grounded set; abort if hallucination/JSON correctness regresses.
- Serve via adapter loading path; preserve ability to swap LLM providers (OpenAI/Claude/Llama). 

