import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

# Allow running the script directly without requiring PYTHONPATH tweaks
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from the project .env file if present.
# Use override=True so a valid key in .env replaces any placeholder set elsewhere.
load_dotenv(PROJECT_ROOT / ".env", override=True)

from src.evaluation.metrics import Evaluator
from src.llm.llm_client import LLMClient
from src.orchestration.chatbot_orchestrator import ChatbotOrchestrator
from src.vector_store.bm25_store import BM25KeywordStore
from src.vector_store.chroma_store import ChromaVectorStore
from src.vector_store.hybrid_retriever import HybridRetriever
from src.vector_store.reranker import CrossEncoderReranker
from src.embeddings.embedding_generator import EmbeddingGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def load_test_cases(path: str = "tests/test_cases.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_orchestrator() -> ChatbotOrchestrator:
    chroma_store = ChromaVectorStore(persist_directory="data/vector_store")
    chroma_store.create_collection(
        collection_name="funnel_drop_chunks",
        embedding_dimension=1536,
        force_recreate=False,
    )

    bm25_store = BM25KeywordStore()
    try:
        bm25_store.load_index("data/bm25_index.pkl")
    except FileNotFoundError:
        logger.warning("BM25 index not found; keyword search will be unavailable until built.")

    embedder = EmbeddingGenerator()
    reranker = CrossEncoderReranker()
    retriever = HybridRetriever(chroma_store, bm25_store, embedder, reranker=reranker)
    llm_client = LLMClient()
    return ChatbotOrchestrator(retriever=retriever, reranker=reranker, llm_client=llm_client)


def run_case(orchestrator: ChatbotOrchestrator, case: Dict) -> Tuple[List[str], Dict]:
    user_state = case.get("user_state", {})
    query = case.get("query", "")

    stage_filter = user_state.get("stage_dropped") or user_state.get("stage")
    metadata_filter = {"stage": stage_filter} if stage_filter else None

    retrieved = orchestrator.retriever.retrieve(
        query,
        top_k=5,
        metadata_filter=metadata_filter,
        strategy="adaptive",
        apply_rerank=True,
        rerank_top_k=10,
    )
    retrieved_ids = [r.get("chunk_id") for r in retrieved if r.get("chunk_id")]

    response = orchestrator.process(user_state, query)
    response["source_ids"] = retrieved_ids
    return retrieved_ids, response


def main():
    test_cases = load_test_cases()
    evaluator = Evaluator()
    orchestrator = init_orchestrator()

    enriched_cases: List[Dict] = []
    responses: List[Dict] = []

    for case in test_cases:
        retrieved_ids, response = run_case(orchestrator, case)
        case_with_retrieved = dict(case)
        case_with_retrieved["retrieved_ids"] = retrieved_ids
        enriched_cases.append(case_with_retrieved)
        responses.append(response)

    relevance = evaluator.evaluate_rag_relevance(enriched_cases)
    hallucination_rate = evaluator.evaluate_hallucination(responses)
    json_correctness = evaluator.evaluate_json_correctness(responses)

    report = {
        "relevance": relevance,
        "hallucination_rate": hallucination_rate,
        "json_correctness": json_correctness,
    }

    Path("docs").mkdir(exist_ok=True)
    Path("docs/evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Evaluation report written to docs/evaluation_report.json")


if __name__ == "__main__":
    main()

