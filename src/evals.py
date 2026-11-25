"""40-Query RAG Evaluation Suite: Before vs. After Reranking, Fallback Rate, Latency & INR Cost."""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any

from src.monitoring import USD_TO_INR
from src.pipeline import EnterpriseRAGPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalQuery:
    """Single evaluation case with expected handbook source and out-of-scope flag."""

    question: str
    expected_source: str
    is_unanswerable: bool = False


# 40 evaluation queries: 35 across the 5 HR handbooks + 5 out-of-scope fallback traps
EVAL_DATASET_40: list[EvalQuery] = [
    # Timelabs_HR_Policy.pdf (8)
    EvalQuery("What are the eligibility requirements for an unpaid leave of absence?", "Timelabs_HR_Policy.pdf"),
    EvalQuery("Who approves an employee's leave of absence request?", "Timelabs_HR_Policy.pdf"),
    EvalQuery("What happens after three consecutive work days of absence without notifying a manager?", "Timelabs_HR_Policy.pdf"),
    EvalQuery("How much advance notice is requested for voluntary resignation?", "Timelabs_HR_Policy.pdf"),
    EvalQuery("Are firearms allowed on Company property?", "Timelabs_HR_Policy.pdf"),
    EvalQuery("What should an employee do if they become ill or have an accident on the job?", "Timelabs_HR_Policy.pdf"),
    EvalQuery("When does an authorized leave for illness or disability begin?", "Timelabs_HR_Policy.pdf"),
    EvalQuery("How can an employee continue benefits during an unpaid leave of absence?", "Timelabs_HR_Policy.pdf"),
    # SHRM_Sample_Handbook.pdf (8)
    EvalQuery("What is the company policy on equal employment opportunity and anti-harassment?", "SHRM_Sample_Handbook.pdf"),
    EvalQuery("How are overtime hours calculated and approved for non-exempt employees?", "SHRM_Sample_Handbook.pdf"),
    EvalQuery("What is the policy regarding FMLA family and medical leave eligibility?", "SHRM_Sample_Handbook.pdf"),
    EvalQuery("How does the progressive discipline process work?", "SHRM_Sample_Handbook.pdf"),
    EvalQuery("What is the policy on lactation accommodations for nursing mothers?", "SHRM_Sample_Handbook.pdf"),
    EvalQuery("Are employees paid for jury duty leave?", "SHRM_Sample_Handbook.pdf"),
    EvalQuery("What are the rules on bereavement leave?", "SHRM_Sample_Handbook.pdf"),
    EvalQuery("What is the policy on meal and rest breaks during a shift?", "SHRM_Sample_Handbook.pdf"),
    # PublicCounsel_HR_Handbook.pdf (7)
    EvalQuery("What is the introductory or probationary period for new hires?", "PublicCounsel_HR_Handbook.pdf"),
    EvalQuery("How are personnel records accessed or updated by employees?", "PublicCounsel_HR_Handbook.pdf"),
    EvalQuery("What is the policy on outside employment or conflicts of interest?", "PublicCounsel_HR_Handbook.pdf"),
    EvalQuery("How are expense reimbursements submitted and approved?", "PublicCounsel_HR_Handbook.pdf"),
    EvalQuery("What is the confidentiality and non-disclosure policy for organizational data?", "PublicCounsel_HR_Handbook.pdf"),
    EvalQuery("What is the whistleblower or open-door complaint reporting procedure?", "PublicCounsel_HR_Handbook.pdf"),
    EvalQuery("How are performance evaluations conducted?", "PublicCounsel_HR_Handbook.pdf"),
    # Valve_Employee_Handbook.pdf (6)
    EvalQuery("How is Valve's organizational structure and hierarchy described?", "Valve_Employee_Handbook.pdf"),
    EvalQuery("How do employees choose which projects to work on at Valve?", "Valve_Employee_Handbook.pdf"),
    EvalQuery("How does peer review and stack ranking work at Valve?", "Valve_Employee_Handbook.pdf"),
    EvalQuery("What is Valve's philosophy on hiring new people?", "Valve_Employee_Handbook.pdf"),
    EvalQuery("How are company-wide vacations or trips handled at Valve?", "Valve_Employee_Handbook.pdf"),
    EvalQuery("What happens during an employee's first week at Valve?", "Valve_Employee_Handbook.pdf"),
    # Workable_Employee_Handbook.pdf (6)
    EvalQuery("What is the remote work and telecommuting policy?", "Workable_Employee_Handbook.pdf"),
    EvalQuery("What is the dress code policy in the workplace?", "Workable_Employee_Handbook.pdf"),
    EvalQuery("What is the policy on workplace drug, alcohol, and smoking use?", "Workable_Employee_Handbook.pdf"),
    EvalQuery("How are company laptops, email, and IT security monitored?", "Workable_Employee_Handbook.pdf"),
    EvalQuery("What is the social media policy for employees?", "Workable_Employee_Handbook.pdf"),
    EvalQuery("How is parental or maternity leave structured?", "Workable_Employee_Handbook.pdf"),
    # Out-of-scope "I don't know" fallback tests (5)
    EvalQuery("What is the company's Dogecoin and cryptocurrency salary payout policy?", "NONE", is_unanswerable=True),
    EvalQuery("How many free flights to Mars does the company sponsor per year?", "NONE", is_unanswerable=True),
    EvalQuery("What is the exact Wi-Fi password for the 4th floor cafeteria?", "NONE", is_unanswerable=True),
    EvalQuery("Does the company reimburse personal scuba diving vacations in Bali?", "NONE", is_unanswerable=True),
    EvalQuery("What is the corporate discount code for Tesla Cybertruck purchases?", "NONE", is_unanswerable=True),
]


def _get_baseline_vector_sources(retriever: Any, question: str, k: int = 4) -> list[str]:
    """Fetch top-k dense vector source names from the retriever's underlying dense store/ensemble."""
    candidate_names = ("vectorstore", "_vectorstore", "vector_store", "_vector_store", "db", "chroma")

    for attr in candidate_names:
        store = getattr(retriever, attr, None)
        if store is not None and hasattr(store, "similarity_search"):
            docs = store.similarity_search(question, k=k)
            return [
                str((getattr(doc, "metadata", {}) or {}).get("source", ""))
                for doc in docs
            ]

    ensemble = getattr(retriever, "ensemble_retriever", None)
    retrievers = getattr(ensemble, "retrievers", None) or []
    for sub_retriever in retrievers:
        if hasattr(sub_retriever, "vectorstore") and hasattr(sub_retriever.vectorstore, "similarity_search"):
            docs = sub_retriever.vectorstore.similarity_search(question, k=k)
            return [
                str((getattr(doc, "metadata", {}) or {}).get("source", ""))
                for doc in docs
            ]
        if hasattr(sub_retriever, "invoke"):
            docs = sub_retriever.invoke(question)[:k]
            return [
                str((getattr(doc, "metadata", {}) or {}).get("source", ""))
                for doc in docs
            ]

    return []


def run_40_query_eval(pipeline: EnterpriseRAGPipeline | None = None) -> dict[str, Any]:
    """Run all 40 evaluation queries and report before/after reranking accuracy, fallback, latency & INR cost."""
    rag = pipeline or EnterpriseRAGPipeline()

    # Clear semantic cache so each query measures real retrieval + LLM latency and cost
    rag.cache.clear()

    vector_only_hits = 0
    reranked_hits = 0
    answerable_total = 0

    fallback_passes = 0
    fallback_total = 0

    latencies_sec: list[float] = []
    costs_inr: list[float] = []

    for idx, item in enumerate(EVAL_DATASET_40, start=1):
        # Step 1: Before-reranking baseline source check
        raw_sources = _get_baseline_vector_sources(rag.retriever, item.question, k=4)

        # Step 2: Run full pipeline and time it
        t0 = time.perf_counter()
        result = rag.run(question=item.question, session_id="eval-40")
        elapsed_sec = round(time.perf_counter() - t0, 3)

        answer_str = str(result.get("answer", ""))
        answer_lower = answer_str.lower()
        model_used = str(result.get("model_used", "")).lower()
        final_sources = [str(s.get("source", "")) for s in (result.get("sources") or [])]
        cost_inr = round(float(result.get("cost_inr", result.get("cost_usd", 0.0) * USD_TO_INR)), 6)

        latencies_sec.append(elapsed_sec)
        costs_inr.append(cost_inr)

        # If a TPD limit / internal error occurred, pause 60s before continuing
        if model_used == "error" or "tokens per day" in answer_lower or "tpd" in answer_lower:
            logger.warning("TPD limit or error detected at query %d/%d; pausing for 60s...", idx, len(EVAL_DATASET_40))
            time.sleep(60.0)

        # Step 3: Score either out-of-scope abstention/guardrail or answerable retrieval accuracy
        if item.is_unanswerable:
            fallback_total += 1
            abstained_or_blocked = (
                "i don't know" in answer_lower
                or "do not contain" in answer_lower
                or "only answers questions related to hr policies" in answer_lower
                or model_used in {"guardrail-blocked", "idk-fallback"}
            )
            if abstained_or_blocked:
                fallback_passes += 1
        else:
            answerable_total += 1
            if raw_sources and any(item.expected_source in s for s in raw_sources):
                vector_only_hits += 1
            if any(item.expected_source in s for s in final_sources):
                reranked_hits += 1

        if idx < len(EVAL_DATASET_40):
            time.sleep(3.0)

    # Step 4: Aggregate headline metrics for resume and telemetry reporting
    before_acc = round((vector_only_hits / max(answerable_total, 1)) * 100, 1)
    after_acc = round((reranked_hits / max(answerable_total, 1)) * 100, 1)
    fallback_acc = round((fallback_passes / max(fallback_total, 1)) * 100, 1)
    median_latency_s = round(statistics.median(latencies_sec), 2) if latencies_sec else 0.0
    avg_cost_inr = round(statistics.mean(costs_inr), 4) if costs_inr else 0.0

    summary = {
        "total_queries": len(EVAL_DATASET_40),
        "before_reranking_accuracy_pct": before_acc,
        "after_reranking_accuracy_pct": after_acc,
        "fallback_accuracy_pct": fallback_acc,
        "median_latency_sec": median_latency_s,
        "avg_cost_per_query_inr": avg_cost_inr,
    }
    print("\n=== 40-QUERY EVALUATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_40_query_eval()