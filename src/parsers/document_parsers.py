import re
from typing import Any, Dict, List


def parse_funnel_steps(text: str) -> List[Dict[str, Any]]:
    funnel_section = re.search(r"1\. Onboarding Funnel:.*?(?=2\. Customer Offers|$)", text, re.DOTALL)
    if not funnel_section:
        return []

    steps_data = [
        {
            "step_number": 1,
            "step_name": "eKYC",
            "process": "Customer provides Aadhaar card number for verification",
            "purpose": "Verification: Checks user identity. Users with existing CSB Bank account can skip.",
            "stage_code": "eKYC",
        },
        {
            "step_number": 2,
            "step_name": "Liveliness",
            "process": "User is required to take a short video of themselves",
            "purpose": "Security: Confirms the user is a real human, not a bot",
            "stage_code": "Liveliness",
        },
        {
            "step_number": 3,
            "step_name": "Additional Details",
            "process": "User fills in mandatory personal details (Father's Name, Mother's Name)",
            "purpose": "Data Collection: Completes required customer profile information",
            "stage_code": "Additional_Details",
        },
        {
            "step_number": 4,
            "step_name": "VKYC Approval",
            "process": "Bank review with Agent video call and Auditor review",
            "purpose": "Approval: Ensures compliance and validity of VKYC",
            "stage_code": "VKYC",
        },
        {
            "step_number": 5,
            "step_name": "OTP",
            "process": "User enters One-Time Password to complete final activation",
            "purpose": "Final Confirmation: The last step to activate the card",
            "stage_code": "OTP",
        },
    ]
    return steps_data


def classify_faq_stage(question: str, answer: str) -> str:
    stage_keywords = {
        "eKYC": ["ekyc", "aadhaar", "existing csb"],
        "Liveliness": ["liveliness", "video of themselves"],
        "VKYC": ["vkyc", "video kyc", "pan card", "agent", "auditor"],
        "OTP": ["otp", "activation", "activate"],
        "General": ["credit limit", "annual fee", "joining fee", "times prime"],
    }
    text = (question + " " + answer).lower()
    for stage, keywords in stage_keywords.items():
        if any(kw in text for kw in keywords):
            return stage
    return "General"


def classify_faq_topic(question: str) -> str:
    q_lower = question.lower()
    if any(word in q_lower for word in ["fee", "annual", "joining", "charge"]):
        return "fees"
    if any(word in q_lower for word in ["limit", "credit limit"]):
        return "credit_limit"
    if any(word in q_lower for word in ["vkyc", "video kyc"]):
        return "vkyc_issues"
    if any(word in q_lower for word in ["card", "physical", "virtual"]):
        return "card_delivery"
    if any(word in q_lower for word in ["times prime", "offer", "benefit"]):
        return "offers"
    if any(word in q_lower for word in ["upi", "transaction"]):
        return "card_usage"
    return "general"


def parse_faqs(text: str) -> List[Dict[str, Any]]:
    faqs: List[Dict[str, Any]] = []
    qa_pattern = r"Question:\s*(.*?)\s*(?:Answer:|$)(.*?)(?=Question:|Call Starter|$)"
    matches = re.finditer(qa_pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if not question or not answer:
            continue

        stage = classify_faq_stage(question, answer)
        topic = classify_faq_topic(question)

        faqs.append(
            {
                "question": question,
                "answer": answer,
                "stage": stage,
                "topic": topic,
                "metadata": {
                    "section": "faq",
                    "stage_related": stage,
                    "topic_category": topic,
                },
            }
        )
    return faqs


def parse_offers(text: str) -> List[Dict[str, Any]]:
    offers_section = re.search(r"2\. Customer Offers and Benefits.*?(?=Question:|3\.|$)", text, re.DOTALL)
    if not offers_section:
        return []

    offers_text = offers_section.group(0)
    benefits: List[Dict[str, Any]] = []
    benefit_pattern = r"([A-Za-z\s+]+?)\s+(\d+(?:\s*\+\s*\d+)?)\s*(month|ticket)"
    matches = re.finditer(benefit_pattern, offers_text, re.IGNORECASE)

    for match in matches:
        benefit_name = match.group(1).strip()
        duration = match.group(2).strip()
        unit = match.group(3).strip()
        benefits.append(
            {
                "benefit_name": benefit_name,
                "duration": f"{duration} {unit}{'s' if unit == 'month' else ''}",
                "metadata": {"section": "offers", "offer_type": "times_prime"},
            }
        )
    return benefits


def parse_call_starters(text: str) -> List[Dict[str, Any]]:
    call_starters: List[Dict[str, Any]] = []
    pattern = r"Call Starter for users dropping off? at ([^:]+?):(.*?)(?=Call Starter|$)"
    matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        stage = match.group(1).strip()
        template = match.group(2).strip()
        template = re.sub(r"\s+", " ", template)
        call_starters.append(
            {
                "stage": stage.lower().replace(" ", "_"),
                "template": template,
                "metadata": {"section": "call_starters", "stage_related": stage},
            }
        )
    return call_starters


def chunk_long_faq(faq: Dict[str, Any], max_tokens: int = 400) -> List[Dict[str, Any]]:
    from src.utils.text_utils import estimate_tokens

    answer = faq["answer"]
    question = faq["question"]
    answer_tokens = estimate_tokens(answer)

    if answer_tokens <= max_tokens:
        return [faq]

    sentences = re.split(r"(?<=[.!?])\s+", answer)
    chunks: List[Dict[str, Any]] = []
    current_chunk = ""
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            chunks.append(
                {
                    "question": question,
                    "answer": current_chunk.strip(),
                    "stage": faq["stage"],
                    "topic": faq["topic"],
                    "is_partial": True,
                    "part_number": len(chunks) + 1,
                }
            )
            current_chunk = sentence + " "
            current_tokens = sentence_tokens
        else:
            current_chunk += sentence + " "
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(
            {
                "question": question,
                "answer": current_chunk.strip(),
                "stage": faq["stage"],
                "topic": faq["topic"],
                "is_partial": True,
                "part_number": len(chunks) + 1,
            }
        )
    return chunks

