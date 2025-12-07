# Design Decisions, Tradeoffs, Limitations, Future Work

## Decisions
- Retrieval: Hybrid (Chroma vectors + BM25) with cross-encoder rerank to cover both semantic and code/error keyword queries.
- Chunking: Structured chunks by funnel stage/FAQ/offers/call starters with enriched metadata (stage, error codes, tone, actions) to enable filters and rerank signals.
- Guardrails: Regex + optional LLM compliance; low-confidence fallback message is fixed to reduce hallucination risk.
- Orchestration: Intent → retrieve/rerank → reason → nudges → synthesize → guardrail; components injectable to swap LLMs/DBs.

## Tradeoffs
- Cross-encoder rerank adds latency but improves precision; can be disabled for speed.
- OpenAI embeddings/LLM for baseline quality; incurs token cost and external dependency.
- Metadata filters rely on consistent stage tagging; mislabeled chunks can hurt recall.

## Limitations
- Fine-tuning/LoRA not yet executed; only design exists.
- Evaluation recall currently low (0.33) on sampled scenarios—needs tuning of expectations/chunk IDs or retrieval weights.
- Token cost tracking not yet wired through responses; latency is relatively high (~20s in sample run).
- Documentation artifacts are static; no automated diagram refresh.

## Future Work
- Execute LoRA on curated dataset; measure before/after on hallucination, JSON correctness, and nudge scores.
- Improve retrieval (stage-aware boosts, better chunk IDs, glare/angle-specific PAN OCR chunks).
- Add cost/latency instrumentation per request and cache embeddings for frequent queries.
- Expand Hindi coverage and add more compliance-safe templates; add PAN-upload specific nudges (glare/angle, lighting).

