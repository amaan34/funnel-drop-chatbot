import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from api.schemas import ChatRequest, NudgeRequest, UserState
from api.middleware import logging_middleware
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.llm.llm_client import LLMClient
from src.vector_store.bm25_store import BM25KeywordStore
from src.vector_store.build_store import load_chunks
from src.vector_store.chroma_store import ChromaVectorStore
from src.vector_store.hybrid_retriever import HybridRetriever
from src.vector_store.reranker import CrossEncoderReranker
from src.orchestration.chatbot_orchestrator import ChatbotOrchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Ensure environment variables from the project .env are available when running via uvicorn
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

app = FastAPI(title="Funnel Drop Chatbot API")
app.middleware("http")(logging_middleware)

ALLOWED_STAGES = {"eKYC", "VKYC", "OTP", "Liveliness", "Additional_Details", "General"}


def _init_retriever() -> HybridRetriever:
    chroma_store = ChromaVectorStore(persist_directory="data/vector_store")
    chroma_store.create_collection(
        collection_name="funnel_drop_chunks",
        embedding_dimension=1536,
        force_recreate=False,
    )

    bm25_store = BM25KeywordStore()
    bm25_index_path = Path("data/bm25_index.pkl")
    if bm25_index_path.exists():
        try:
            bm25_store.load_index(str(bm25_index_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("BM25 index load failed, will attempt rebuild. Error: %s", exc)
    if not bm25_store.bm25:
        try:
            chunks = load_chunks("data/processed_chunks.json")
            bm25_store.build_index(chunks)
            bm25_store.save_index(str(bm25_index_path))
            logger.info("BM25 index built from processed chunks at startup.")
        except FileNotFoundError:
            logger.error(
                "Processed chunks not found at data/processed_chunks.json; "
                "keyword search will remain unavailable until chunks are built."
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("BM25 auto-build failed: %s", exc)

    embedder = EmbeddingGenerator()
    reranker = CrossEncoderReranker()
    retriever = HybridRetriever(chroma_store, bm25_store, embedder, reranker=reranker)
    return retriever


retriever = _init_retriever()
llm_client = LLMClient()
orchestrator = ChatbotOrchestrator(retriever=retriever, reranker=retriever.reranker, llm_client=llm_client)


def _validate_user_state(user_state: UserState) -> None:
    if user_state.stage_dropped and user_state.stage_dropped not in ALLOWED_STAGES:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {user_state.stage_dropped}")


@app.post("/predict_reason")
def predict_reason(user_state: UserState):
    try:
        _validate_user_state(user_state)
        reasoning = orchestrator.drop_off_reasoner.analyze(user_state.dict(), [])
        return reasoning
    except Exception as exc:  # noqa: BLE001
        logger.exception("predict_reason failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to predict reason") from exc


@app.post("/nudge_user")
def nudge_user(request: NudgeRequest):
    try:
        _validate_user_state(request.user_state)
        reasoning = orchestrator.drop_off_reasoner.analyze(request.user_state.dict(), [])
        nudges = orchestrator.nudge_generator.generate_all(
            reasoning.get("primary_reason", ""),
            user_state=request.user_state.dict(),
            language=request.language,
        )
        return {"reasoning": reasoning, "nudges": nudges}
    except Exception as exc:  # noqa: BLE001
        logger.exception("nudge_user failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate nudge") from exc


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        _validate_user_state(request.user_state)
        response = orchestrator.process(request.user_state.dict(), request.query)
        return response
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed: %s", exc)
        raise HTTPException(status_code=500, detail="Chat processing failed") from exc


@app.get("/health")
def health():
    return {"status": "ok"}

