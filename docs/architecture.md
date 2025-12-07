# Architecture & Data Flow

```mermaid
flowchart TD
    U[User / API Client] -->|/chat| Intent[Intent Classifier (LLM)]
    U -->|/predict_reason| Orchestrator
    U -->|/nudge_user| Orchestrator

    Intent --> Orchestrator[Chatbot Orchestrator]

    Orchestrator --> Retriever[Hybrid Retriever\n(Chroma vectors + BM25)]
    Retriever -->|semantic + keyword| Chunks[Chunk Store\nwith metadata]
    Retriever -->|rerank| Reranker[Cross-Encoder]

    Orchestrator --> Reasoner[Drop-Off Reasoner\n(rules + LLM CoT)]
    Orchestrator --> NudgeGen[Nudge Generator\n(Explanatory/CTA/Empathetic/Compliance-safe, bilingual)]
    Orchestrator --> Synth[Response Synthesizer\n(citations, low-conf guardrail)]
    Synth --> Guard[Guardrail Validator\n(compliance checks)]

    Chunks -.-> DocProc[Document Processing\n(PDF -> Chunks -> Embeddings)]
    DocProc -->|processed_chunks.json| Chunks
    DocProc -->|BM25 index| Retriever
    DocProc -->|Chroma collection| Retriever

    Guard --> Out[Final Response\n(JSON + message + citations)]
```

## Modular swap points
- LLM: `LLMClient` can point to OpenAI/Claude/Llama; LoRA adapters loadable when available.
- Vector DB: `ChromaVectorStore` behind `HybridRetriever`—can be replaced with any DB implementing the same methods.
- Reranker: `CrossEncoderReranker` pluggable; can disable or swap to another model.
- Compliance: `ComplianceValidator` allows regex-only or LLM-assisted checks.


