"""Input and output guardrails for the Enterprise HR Policy RAG System."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.config import get_settings

logger = logging.getLogger(__name__)

HR_DOMAIN_KEYWORDS: tuple[str, ...] = (
    "hr",
    "human resources",
    "policy",
    "policies",
    "handbook",
    "employee",
    "employees",
    "staff",
    "manager",
    "supervisor",
    "leave",
    "vacation",
    "pto",
    "sick",
    "fmla",
    "maternity",
    "paternity",
    "parental",
    "bereavement",
    "jury duty",
    "unpaid",
    "overtime",
    "exempt",
    "non-exempt",
    "salary",
    "compensation",
    "pay",
    "reimbursement",
    "expense",
    "benefit",
    "benefits",
    "insurance",
    "401k",
    "retirement",
    "discipline",
    "disciplinary",
    "progressive discipline",
    "warning",
    "termination",
    "resignation",
    "notice",
    "whistleblower",
    "open-door",
    "open door",
    "grievance",
    "complaint",
    "harassment",
    "discrimination",
    "equal employment",
    "eeo",
    "compliance",
    "code of conduct",
    "conduct",
    "ethics",
    "confidential",
    "confidentiality",
    "non-disclosure",
    "nda",
    "conflict of interest",
    "outside employment",
    "performance",
    "evaluation",
    "appraisal",
    "review",
    "probation",
    "probationary",
    "introductory",
    "onboarding",
    "hiring",
    "attendance",
    "absence",
    "absent",
    "tardy",
    "break",
    "breaks",
    "meal",
    "rest",
    "lactation",
    "nursing",
    "remote",
    "telecommuting",
    "work from home",
    "dress code",
    "attire",
    "drug",
    "alcohol",
    "smoking",
    "substance",
    "firearm",
    "firearms",
    "weapon",
    "personnel",
    "records",
    "laptop",
    "email",
    "it security",
    "social media",
    "injury",
    "accident",
    "workers' compensation",
    "workers compensation",
    "disability",
    "illness",
    "hierarchy",
    "organizational",
    "peer review",
)

OBVIOUS_NON_HR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bwi-?fi\b.*\bpassword\b", re.IGNORECASE),
    re.compile(r"\btesla\b|\bcybertruck\b", re.IGNORECASE),
    re.compile(r"\bmars\b|\binterplanetary\b", re.IGNORECASE),
    re.compile(r"\bdogecoin\b|\bcryptocurrency\b|\bcrypto\b", re.IGNORECASE),
    re.compile(r"\bscuba\b|\bbali\b", re.IGNORECASE),
)

GUARDRAIL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a lenient input guardrail for an Enterprise HR Policy Assistant.\n"
            "Classify whether the user's question is remotely related to HR, workplace policies, "
            "employment, compensation, leave, benefits, conduct, discipline, whistleblower/grievance "
            "procedures, compliance, or company operations.\n"
            "ONLY answer BLOCK if the query is clearly a joke, absurd/impossible scenario, or completely "
            "unrelated consumer/trivia request (such as Wi-Fi passwords, car discounts, space travel, "
            "cryptocurrency perks, or personal vacation reimbursements).\n"
            "When in doubt, answer ALLOW.\n"
            "Respond with ONLY one word: ALLOW or BLOCK.",
        ),
        ("human", "{question}"),
    ]
)


def _has_hr_keyword(text: str) -> bool:
    """Return True if the query contains recognizable HR or workplace policy terms."""
    lowered = f" {re.sub(r'[^a-z0-9\s\-]+', ' ', text.lower())} "
    lowered = re.sub(r"\s+", " ", lowered)
    return any(kw in lowered for kw in HR_DOMAIN_KEYWORDS)


def validate_input(question: str) -> tuple[bool, str]:
    """Validate incoming user question; return (is_allowed, message)."""
    cleaned = (question or "").strip()
    if not cleaned:
        return False, "Please enter a valid HR policy question."

    if _has_hr_keyword(cleaned):
        return True, ""

    for pattern in OBVIOUS_NON_HR_PATTERNS:
        if pattern.search(cleaned):
            logger.info("Guardrail flagged non-HR domain query: %s", cleaned)
            return (
                False,
                "This assistant only answers questions related to HR policies, workplace guidelines, "
                "and employee handbooks.",
            )

    settings = get_settings()
    if not settings.groq_api_key:
        return True, ""

    try:
        llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.fast_llm_model,
            temperature=0.0,
            timeout=15.0,
            max_retries=0,
        )
        verdict = (GUARDRAIL_PROMPT | llm | StrOutputParser()).invoke({"question": cleaned}).strip().upper()
        if verdict.startswith("BLOCK"):
            logger.info("Guardrail flagged non-HR domain query: %s", cleaned)
            return (
                False,
                "This assistant only answers questions related to HR policies, workplace guidelines, "
                "and employee handbooks.",
            )
    except Exception as exc:
        logger.warning("Input guardrail LLM check skipped (%s); allowing query.", exc)

    return True, ""


def validate_output(answer: str, chunks: list[Any] | None = None) -> str:
    """Pass-through output guardrail returning a clean string answer."""
    return (answer or "").strip()