import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.chunking.chunking import ChunkingStrategy, ChunkValidator, MetadataEnricher
from src.extractors.pymupdf_extractor import extract_text_pymupdf
from src.parsers.document_parsers import (
    parse_call_starters,
    parse_faqs,
    parse_funnel_steps,
    parse_offers,
)
from src.utils.text_utils import initial_clean

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


class DocumentChunker:
    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        self.raw_text: Optional[str] = None
        self.structured_data: Dict[str, Any] = {}
        self.chunks: List[Dict[str, Any]] = []

    def process(self) -> List[Dict[str, Any]]:
        logging.info("Step 1: Extracting text from PDF")
        self.raw_text = self.extract_text()

        logging.info("Step 2: Parsing structured content")
        self.structured_data = self.parse_document()

        logging.info("Step 3: Creating semantic chunks")
        self.chunks = self.create_chunks()

        logging.info("Step 4: Enriching metadata")
        self.chunks = self.enrich_all_chunks()

        logging.info("Processing complete: %s chunks created", len(self.chunks))
        return self.chunks

    def extract_text(self) -> str:
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found at {self.pdf_path}")

        extraction_attempts = [("pymupdf", extract_text_pymupdf)]

        last_error: Optional[Exception] = None
        for name, extractor in extraction_attempts:
            try:
                logging.info("Attempting text extraction with %s", name)
                text = extractor(self.pdf_path)
                return initial_clean(text)
            except Exception as exc:  # pragma: no cover - defensive
                last_error = exc
                logging.warning("%s extraction failed: %s", name, exc)

        raise RuntimeError(f"All extraction methods failed: {last_error}")

    def parse_document(self) -> Dict[str, Any]:
        assert self.raw_text is not None, "Raw text must be extracted before parsing"
        return {
            "funnel_steps": parse_funnel_steps(self.raw_text),
            "faqs": parse_faqs(self.raw_text),
            "offers": parse_offers(self.raw_text),
            "call_starters": parse_call_starters(self.raw_text),
        }

    def create_chunks(self) -> List[Dict[str, Any]]:
        strategy = ChunkingStrategy()
        all_chunks: List[Dict[str, Any]] = []
        all_chunks.extend(strategy.chunk_funnel_steps(self.structured_data.get("funnel_steps", [])))
        all_chunks.extend(strategy.chunk_faqs(self.structured_data.get("faqs", [])))
        all_chunks.extend(strategy.chunk_offers(self.structured_data.get("offers", [])))
        all_chunks.extend(strategy.chunk_call_starters(self.structured_data.get("call_starters", [])))
        return all_chunks

    def enrich_all_chunks(self) -> List[Dict[str, Any]]:
        enricher = MetadataEnricher()
        enriched: List[Dict[str, Any]] = []
        for chunk in self.chunks:
            enriched.append(enricher.enrich_chunk(chunk, self.structured_data))
        return enriched

    def save_chunks(self, output_path: Path = Path("data/processed_chunks.json")) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        logging.info("Chunks saved to %s", output_path)

    def get_statistics(self) -> Dict[str, Any]:
        if not self.chunks:
            return {"total_chunks": 0, "by_section": {}, "by_stage": {}, "avg_chunk_size": 0}

        stats: Dict[str, Any] = {
            "total_chunks": len(self.chunks),
            "by_section": {},
            "by_stage": {},
            "avg_chunk_size": sum(c["metadata"]["chunk_length"] for c in self.chunks) / len(self.chunks),
        }

        for chunk in self.chunks:
            section = chunk["metadata"].get("section", "unknown")
            stage = chunk["metadata"].get("stage", "general")
            stats["by_section"][section] = stats["by_section"].get(section, 0) + 1
            stats["by_stage"][stage] = stats["by_stage"].get(stage, 0) + 1
        return stats


def main(
    pdf_path: str = "data/Assignment RAG_context_file.pdf",
    output_path: str = "data/processed_chunks.json",
) -> List[Dict[str, Any]]:
    chunker = DocumentChunker(Path(pdf_path))
    chunks = chunker.process()

    stats = chunker.get_statistics()
    logging.info("Total chunks: %s", stats["total_chunks"])
    logging.info("Average chunk size: %.0f characters", stats["avg_chunk_size"])
    logging.info("Chunks by section: %s", stats["by_section"])
    logging.info("Chunks by stage: %s", stats["by_stage"])

    validator = ChunkValidator()
    issues = validator.validate_chunks(chunks)
    validator.print_validation_report(issues)

    chunker.save_chunks(Path(output_path))
    for i, chunk in enumerate(chunks[:3]):
        logging.info("Sample chunk %s (%s): %s", i + 1, chunk["chunk_id"], chunk["content"][:200])
    return chunks


if __name__ == "__main__":  # pragma: no cover - manual run helper
    main()

