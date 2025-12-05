from typing import Any, Dict, List

from src.parsers.document_parsers import chunk_long_faq
from src.utils.text_utils import (
    detect_tone,
    estimate_tokens,
    extract_error_codes,
    extract_timeline_mentions,
    get_stage_priority,
    has_actionable_steps,
    normalize_stage_name,
)


class ChunkingStrategy:
    @staticmethod
    def chunk_funnel_steps(steps_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for step in steps_data:
            chunk_content = (
                f"Onboarding Funnel Step {step['step_number']}: {step['step_name']}\n"
                f"Process: {step['process']}\n"
                f"Purpose: {step['purpose']}\n"
                f"Stage Code: {step['stage_code']}"
            )
            chunks.append(
                {
                    "content": chunk_content,
                    "metadata": {
                        "section": "onboarding_funnel",
                        "stage": step["stage_code"],
                        "step_number": step["step_number"],
                        "chunk_type": "funnel_step",
                        "stage_name": step["step_name"],
                    },
                    "chunk_id": f"funnel_step_{step['step_number']}",
                }
            )
        return chunks

    @staticmethod
    def chunk_faqs(faqs_data: List[Dict[str, Any]], max_tokens: int = 400) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for idx, faq in enumerate(faqs_data):
            faq_parts = chunk_long_faq(faq, max_tokens=max_tokens)
            for part_idx, part in enumerate(faq_parts):
                chunk_content = (
                    f"Question: {part['question']}\n\n"
                    f"Answer: {part['answer']}\n\n"
                    f"Related Stage: {part['stage']}\n"
                    f"Topic Category: {part['topic']}"
                )
                chunk_id_suffix = f"_part{part_idx + 1}" if len(faq_parts) > 1 else ""
                chunks.append(
                    {
                        "content": chunk_content,
                        "metadata": {
                            "section": "faq",
                            "stage": part["stage"],
                            "topic": part["topic"],
                            "chunk_type": "faq",
                            "question": part["question"][:100],
                            "is_partial": part.get("is_partial", False),
                            "part_number": part.get("part_number"),
                        },
                        "chunk_id": f"faq_{idx}{chunk_id_suffix}",
                    }
                )
        return chunks

    @staticmethod
    def chunk_offers(offers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        if not offers_data:
            return chunks

        summary = "Times Prime Membership Benefits (12-month Premium Package, MRP ₹1299):\n\n"
        for offer in offers_data:
            summary += f"- {offer['benefit_name']}: {offer['duration']}\n"

        chunks.append(
            {
                "content": summary.strip(),
                "metadata": {
                    "section": "offers",
                    "offer_type": "times_prime",
                    "chunk_type": "offers_summary",
                },
                "chunk_id": "offers_times_prime",
            }
        )
        return chunks

    @staticmethod
    def chunk_call_starters(call_starters_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for starter in call_starters_data:
            chunk_content = (
                f"Call Starter Template for {starter['stage']} Stage:\n\n"
                f"{starter['template']}\n\n"
                f"Use Case: Reaching out to users who dropped off at {starter['stage']} stage"
            )
            chunks.append(
                {
                    "content": chunk_content,
                    "metadata": {
                        "section": "call_starters",
                        "stage": starter["stage"],
                        "chunk_type": "call_template",
                    },
                    "chunk_id": f"call_starter_{starter['stage']}",
                }
            )
        return chunks


class MetadataEnricher:
    @staticmethod
    def enrich_chunk(chunk: Dict[str, Any], document_context: Dict[str, Any]) -> Dict[str, Any]:
        metadata = chunk.get("metadata", {})
        metadata.update(
            {
                "document_source": "credit_card_onboarding_faq",
                "document_version": "1.0",
                "chunk_id": chunk["chunk_id"],
                "chunk_length": len(chunk["content"]),
                "chunk_tokens": estimate_tokens(chunk["content"]),
            }
        )

        if "stage" in metadata and metadata["stage"]:
            metadata["stage_canonical"] = normalize_stage_name(str(metadata["stage"]))
            metadata["stage_priority"] = get_stage_priority(str(metadata["stage"]))

        error_codes = extract_error_codes(chunk["content"])
        if error_codes:
            metadata["error_codes_mentioned"] = error_codes

        content_lower = chunk["content"].lower()
        metadata["mentions_credit_limit"] = "credit limit" in content_lower
        metadata["mentions_fees"] = any(word in content_lower for word in ["fee", "charge", "cost"])
        timelines = extract_timeline_mentions(chunk["content"])
        if timelines:
            metadata["mentions_timeline"] = timelines

        metadata["tone"] = detect_tone(chunk["content"])
        metadata["has_action_steps"] = has_actionable_steps(chunk["content"])
        chunk["metadata"] = metadata
        return chunk


class ChunkValidator:
    @staticmethod
    def validate_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
        issues: List[str] = []
        for idx, chunk in enumerate(chunks):
            if not chunk.get("content") or len(chunk["content"]) < 20:
                issues.append(f"Chunk {idx}: Content too short or missing")
            if not chunk.get("metadata"):
                issues.append(f"Chunk {idx}: Missing metadata")
            if not chunk.get("chunk_id"):
                issues.append(f"Chunk {idx}: Missing chunk_id")
            if chunk.get("metadata", {}).get("chunk_tokens", 0) > 600:
                issues.append(f"Chunk {idx}: Too large ({chunk['metadata']['chunk_tokens']} tokens)")

            valid_sections = ["onboarding_funnel", "faq", "offers", "call_starters"]
            section = chunk.get("metadata", {}).get("section")
            if section not in valid_sections:
                issues.append(f"Chunk {idx}: Invalid section type '{section}'")

        chunk_ids = [c["chunk_id"] for c in chunks]
        duplicates = [cid for cid in chunk_ids if chunk_ids.count(cid) > 1]
        if duplicates:
            issues.append(f"Duplicate chunk_ids found: {set(duplicates)}")
        return issues

    @staticmethod
    def print_validation_report(issues: List[str]) -> None:
        if not issues:
            return
        import logging

        logging.warning("Validation issues detected:")
        for issue in issues:
            logging.warning(" - %s", issue)

