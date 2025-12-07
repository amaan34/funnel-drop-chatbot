System architecture (chatbot)
=============================

```mermaid
graph TD
    U[User / Client] -->|REST/JSON| API[FastAPI main.py]
    API --> MW[Middleware & Auth]
    MW --> ORCH[ChatbotOrchestrator]

    ORCH --> INTENT[IntentClassifier]
    ORCH --> RETRIEVE[HybridRetriever]
    RETRIEVE -->|semantic| CHROMA[Chroma vector store]
    RETRIEVE -->|keyword| BM25[BM25 index]
    RETRIEVE --> RERANK[Cross-encoder reranker]

    ORCH --> REASON[DropOffReasoner]
    ORCH --> CONF[ConfidenceScorer]
    ORCH --> NUDGE[NudgeGenerator]
    ORCH --> SYNTH[ResponseSynthesizer]
    ORCH --> GUARD[GuardrailValidator]

    CHROMA -. builds from .-> DOCS[data/processed_chunks.json]
    BM25 -. builds from .-> DOCS

    RERANK --> REASON
    RETRIEVE --> REASON
    REASON --> SYNTH
    CONF --> SYNTH
    NUDGE --> SYNTH
    SYNTH --> GUARD
    GUARD --> API
    API --> U
```

Data flow highlights
--------------------
- Hybrid retrieval (Chroma + BM25) feeds reranker, then reasoning + nudge generation.
- Guardrails wrap final responses; citations validated before returning to the client.

