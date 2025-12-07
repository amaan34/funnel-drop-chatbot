Evaluation summary (2025-12-06)
===============================

- Scenario set expanded to 6 cases: VKYC OCR, OTP timeout, doc checklist, PAN glare/angle, VKYC hours/window, OTP retry limits.
- Key metrics: recall@5=0.83 (5/6 hits), hallucination=0.00, JSON correctness=1.00, avg latency=0.98s, avg token cost=0.0045, nudge helpfulness=4.27 (n=6).
- Token cost now captured from responses; use as baseline for cost regression checks.
- Recall labels tightened with explicit expected chunk IDs for new edge cases (PAN glare, VKYC window, OTP retry) to pressure-test retrieval.
- No LLM judge was invoked in this offline run; scores are pre-labeled. Re-run with judge to validate alignment before release.
- See `data/eval_report.json` for the full structured output.

