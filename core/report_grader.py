# core/report_grader.py
import re
from core.llm_utils import call_llm  # ✅ 공용 LLM 호출 유틸 사용

def grade_report_quality(state: dict) -> str:
    """
    생성된 보고서의 품질을 평가하는 함수 (Qwen 기반)
    """
    report = state.get("report") or state.get("candidate_answer", "")
    if not report:
        return "insufficient"

    question = (
        "다음 건설안전 보고서가 충분히 완전한가? "
        "주요 항목(사고 개요, 위험 요인, 즉시 조치, 관련 규정)이 모두 다뤄졌는지 평가하라. "
        "부족하면 'insufficient', 충분하면 'adequate'로만 JSON 형식으로 출력하라. "
        "예시: {\"verdict\": \"adequate\"}"
    )

    # ✅ LLM 호출 (공용 함수 사용)
    messages = [
        {"role": "system", "content": "당신은 건설안전 보고서 품질 평가자입니다."},
        {"role": "user", "content": f"{question}\n\n보고서:\n{report}"}
    ]
    raw = call_llm(messages)

    # ✅ 결과 파싱
    if re.search(r"adequate", raw.lower()):
        verdict = "adequate"
    elif re.search(r"insufficient", raw.lower()):
        verdict = "insufficient"
    else:
        verdict = "insufficient"

    print(f"🧾 보고서 품질 평가 결과: {verdict.upper()}")
    return verdict
